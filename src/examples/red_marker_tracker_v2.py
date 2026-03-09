"""
Robust Red Marker Tracker for Robotic Structure
================================================
Tracks 13 red markers on a robotic structure subjected to impulse excitation.

OUTPUT FILES
------------
  trajectories.h5  — HDF5 file with hierarchy:
                       time_series/
                           time          (F,)       timestamps in seconds
                           nodes/
                               positions (F, N, 2)  x/y per frame per marker
                               node_ids  (N,)        marker IDs 0..NUM_MARKERS-1

Edit the CONFIG block below to set paths and detection parameters.

Hotkeys during live preview:
    Q  —  quit (data saved on exit)
    S  —  save screenshot of current quad view
    R  —  reset all trackers (clears trajectory history)
    T  —  open interactive HSV tuner
"""

import cv2
import numpy as np
import sys
import h5py
from pathlib import Path
from collections import deque
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import time


# ═══════════════════════════════════════════════════════════════
#  CONFIG  —  edit everything here
# ═══════════════════════════════════════════════════════════════

# Input: path to a video file (str) or webcam index (int, e.g. 0)


# HDF5 hierarchy written:
#   time_series/
#       time          (F,)        timestamps in seconds  (float64)
#       nodes/
#           positions (F, N, 2)   x/y per frame per marker (float64, NaN=lost)
#           node_ids  (N,)        marker IDs 0..NUM_MARKERS-1 (int32)

# Video frame rate — exact NTSC 29.97 = 30000/1001
VIDEO_FPS = 30000.0 / 1001.0      # 29.970029… Hz

# Live preview window (set SHOW_PREVIEW = False for headless/faster processing)
SHOW_PREVIEW = True    # show the quad-view OpenCV window while processing
QUAD_W       = 1280    # quad-view width  (px)
QUAD_H       = 720     # quad-view height (px)

# Marker detection
NUM_MARKERS = 19      # expected number of markers
MIN_AREA    = 80      # minimum contour area (px2)
MAX_AREA    = 8000    # maximum contour area (px2)
MAX_DIST    = 180     # max Hungarian-match distance (px)
MAX_LOST    = 15      # frames before a tracker is marked inactive

# Initialisation
INIT_FRAMES = 1       # frames of stable video to observe before locking IDs
                      # increase if the structure is already moving at frame 0

# Pre-processing
BLUR_KERNEL = 5       # Gaussian blur kernel (must be odd)
MORPH_SIZE  = 7       # morphological open/close kernel size

# Visualisation (only affects the live preview, not saved data)
SHOW_IDS = True

# HSV colour thresholds for red (two ranges needed — red wraps at 0/180 in HSV)
# Press T during preview to tune interactively, then paste printed values here.
RED_LOWER_1 = (0,   75,  80)
RED_UPPER_1 = (10,  255, 255)
RED_LOWER_2 = (165, 75,  80)
RED_UPPER_2 = (180, 255, 255)

# ═══════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────
#  Marker data structure
# ─────────────────────────────────────────────

@dataclass
class Marker:
    """Single tracked marker with Kalman prediction and position history."""
    marker_id:   int
    centroid:    Tuple[float, float]
    area:        float
    bbox:        Tuple[int, int, int, int]
    contour:     np.ndarray
    history:     deque = field(default_factory=lambda: deque(maxlen=60))
    lost_frames: int   = 0
    active:      bool  = True
    kalman:      Optional[cv2.KalmanFilter] = None

    def __post_init__(self):
        self.history.append(self.centroid)
        self.kalman = self._init_kalman(self.centroid)

    def _init_kalman(self, centroid: Tuple[float, float]) -> cv2.KalmanFilter:
        """
        4-state Kalman filter [x, y, vx, vy].
        High process noise on velocity adapts quickly to impulsive motion.
        """
        kf = cv2.KalmanFilter(4, 2)
        dt = 1.0
        kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float32)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        kf.processNoiseCov[2, 2] = 5.0
        kf.processNoiseCov[3, 3] = 5.0
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        kf.statePost = np.array(
            [centroid[0], centroid[1], 0.0, 0.0], dtype=np.float32
        ).reshape(4, 1)
        return kf

    def predict(self) -> Tuple[float, float]:
        pred = self.kalman.predict()
        return float(pred[0][0]), float(pred[1][0])

    def correct(self, centroid: Tuple[float, float]):
        meas = np.array([[centroid[0]], [centroid[1]]], dtype=np.float32)
        self.kalman.correct(meas)
        self.centroid = centroid
        self.history.append(centroid)
        self.lost_frames = 0
        self.active = True

    def update_lost(self):
        pred = self.predict()
        self.centroid = pred
        self.history.append(pred)
        self.lost_frames += 1


# ─────────────────────────────────────────────
#  Detection helpers
# ─────────────────────────────────────────────

def build_hsv_mask(hsv: np.ndarray, morph_kernel: np.ndarray) -> np.ndarray:
    lo1 = np.array(RED_LOWER_1, dtype=np.uint8)
    hi1 = np.array(RED_UPPER_1, dtype=np.uint8)
    lo2 = np.array(RED_LOWER_2, dtype=np.uint8)
    hi2 = np.array(RED_UPPER_2, dtype=np.uint8)
    mask = cv2.bitwise_or(cv2.inRange(hsv, lo1, hi1),
                          cv2.inRange(hsv, lo2, hi2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  morph_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph_kernel, iterations=2)
    return mask


def detect_candidates(mask: np.ndarray,
                      min_area: float,
                      max_area: float) -> List[Dict]:
    """Return contours that pass area + circularity filters."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        x, y, w, h = cv2.boundingRect(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        if 4 * np.pi * area / (perimeter * perimeter) < 0.1:
            continue
        candidates.append({
            "centroid": (cx, cy),
            "area":     area,
            "bbox":     (x, y, w, h),
            "contour":  cnt,
        })
    return candidates


# ─────────────────────────────────────────────
#  Hungarian assignment
# ─────────────────────────────────────────────

def _cost_matrix(trackers, detections, max_dist):
    n, m = len(trackers), len(detections)
    C = np.full((n, m), fill_value=max_dist * 10, dtype=np.float64)
    for i, tr in enumerate(trackers):
        px, py = tr.predict()
        for j, det in enumerate(detections):
            dx, dy = det["centroid"]
            d = np.hypot(px - dx, py - dy)
            if d < max_dist:
                C[i, j] = d
    return C


def hungarian_match(trackers, detections, max_dist):
    if not trackers or not detections:
        return [], list(range(len(trackers))), list(range(len(detections)))
    C = _cost_matrix(trackers, detections, max_dist)
    row_ind, col_ind = linear_sum_assignment(C)
    matched, assigned_t, assigned_d = [], set(), set()
    for r, c in zip(row_ind, col_ind):
        if C[r, c] < max_dist:
            matched.append((r, c))
            assigned_t.add(r)
            assigned_d.add(c)
    unmatched_t = [i for i in range(len(trackers))   if i not in assigned_t]
    unmatched_d = [j for j in range(len(detections)) if j not in assigned_d]
    return matched, unmatched_t, unmatched_d


# ─────────────────────────────────────────────
#  Visualisation helpers (preview only)
# ─────────────────────────────────────────────

def draw_quad_view(original, mask, overlay, black_bg, target_size=(1280, 720)):
    tw, th = target_size
    hw, hh = tw // 2, th // 2
    def rsz(img, w, h):
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    q1 = rsz(original, hw, hh)
    q2 = rsz(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), hw, hh)
    q3 = rsz(overlay,  hw, hh)
    q4 = rsz(black_bg, hw, hh)
    return np.vstack([np.hstack([q1, q2]), np.hstack([q3, q4])])


def add_labels(canvas, target_size=(1280, 720)):
    tw, th = target_size
    hw, hh = tw // 2, th // 2
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    pad = 8
    for (x, y), text in [
        ((pad,      hh - pad), "Original"),
        ((hw + pad, hh - pad), "HSV Mask"),
        ((pad,      th - pad), "Tracked (overlay)"),
        ((hw + pad, th - pad), "Tracked (black bg)"),
    ]:
        cv2.putText(canvas, text, (x, y), font, scale, (0, 0, 0),      thick + 1)
        cv2.putText(canvas, text, (x, y), font, scale, (200, 200, 200), thick)
    cv2.line(canvas, (hw, 0),  (hw, th), (60, 60, 60), 1)
    cv2.line(canvas, (0,  hh), (tw, hh), (60, 60, 60), 1)


# ─────────────────────────────────────────────
#  Interactive HSV tuner
# ─────────────────────────────────────────────

def hsv_tuner(cap: cv2.VideoCapture):
    WIN = "HSV Tuner  (Q = done)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 900, 600)
    def nothing(_): pass
    for name, default, mx in [
        ("H_lo1",  0,   30),  ("S_lo1", 120, 255), ("V_lo1",  80, 255),
        ("H_hi1",  10,  30),  ("S_hi1", 255, 255), ("V_hi1", 255, 255),
        ("H_lo2", 165, 180),  ("S_lo2", 120, 255), ("V_lo2",  80, 255),
        ("H_hi2", 180, 180),  ("S_hi2", 255, 255), ("V_hi2", 255, 255),
    ]:
        cv2.createTrackbar(name, WIN, default, mx, nothing)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        def tb(n): return cv2.getTrackbarPos(n, WIN)
        lo1 = np.array([tb("H_lo1"), tb("S_lo1"), tb("V_lo1")], dtype=np.uint8)
        hi1 = np.array([tb("H_hi1"), tb("S_hi1"), tb("V_hi1")], dtype=np.uint8)
        lo2 = np.array([tb("H_lo2"), tb("S_lo2"), tb("V_lo2")], dtype=np.uint8)
        hi2 = np.array([tb("H_hi2"), tb("S_hi2"), tb("V_hi2")], dtype=np.uint8)
        mask = cv2.bitwise_or(cv2.inRange(hsv, lo1, hi1), cv2.inRange(hsv, lo2, hi2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        vis  = np.hstack([cv2.resize(frame, (450, 400)),
                          cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (450, 400))])
        cv2.imshow(WIN, vis)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("\n--- Copy these values into the CONFIG block ---")
            print(f"RED_LOWER_1 = {tuple(lo1.tolist())}")
            print(f"RED_UPPER_1 = {tuple(hi1.tolist())}")
            print(f"RED_LOWER_2 = {tuple(lo2.tolist())}")
            print(f"RED_UPPER_2 = {tuple(hi2.tolist())}")
            break
    cv2.destroyWindow(WIN)


# ─────────────────────────────────────────────
#  Trajectory data store
# ─────────────────────────────────────────────

class TrajectoryStore:
    """
    Accumulates per-frame centroid data and saves to HDF5.

    HDF5 hierarchy:
        time_series/
            time          (F,)       timestamps in seconds (float64)
            nodes/
                positions (F, N, 2)  x/y centroid per frame per marker (float64)
                                     NaN where marker was lost/occluded
                node_ids  (N,)       marker IDs 0..NUM_MARKERS-1 (int32)

    Root attributes: source_file, video_fps, num_markers, num_frames
    """

    def __init__(self, num_markers: int, video_fps: float):
        self.num_markers = num_markers
        self.video_fps   = video_fps
        self._frame_ids: List[int]   = []
        self._times:     List[float] = []
        # _pos[frame_index][marker_id] = (x, y)  or  None when lost
        self._pos: List[List[Optional[Tuple[float, float]]]] = []

    def record(self, frame_idx: int, time_s: float, trackers: List[Marker]):
        """Store one frame. Only called after ID lock so IDs are 0..N-1."""
        self._frame_ids.append(frame_idx)
        self._times.append(time_s)
        row: List[Optional[Tuple[float, float]]] = [None] * self.num_markers
        for t in trackers:
            if 0 <= t.marker_id < self.num_markers and t.active:
                row[t.marker_id] = t.centroid
            # lost markers → NaN (leave row entry as None)
        self._pos.append(row)

    # ── HDF5 export ───────────────────────────────────────────

    def save_h5(self, SOURCE, path: str):
        """
        Write HDF5 with the structure:

            time_series/
                time          (F,)      seconds, float64
                nodes/
                    positions (F, N, 2) pixels, float64 — NaN = lost
                    node_ids  (N,)      int32, values 0..N-1
        """
        F = len(self._frame_ids)
        N = self.num_markers

        if F == 0:
            print("[WARN] No frames recorded — HDF5 not written.")
            return

        # Build arrays
        time_arr = np.array(self._times, dtype=np.float64)        # (F,)
        pos_arr  = np.full((F, N, 2), np.nan, dtype=np.float64)   # (F, N, 2)
        for fi, row in enumerate(self._pos):
            for mid, xy in enumerate(row):
                if xy is not None:
                    pos_arr[fi, mid, 0] = xy[0]   # x  (horizontal, left→right)
                    pos_arr[fi, mid, 1] = xy[1]   # y  (vertical,   top→bottom)
        node_ids = np.arange(N, dtype=np.int32)                   # (N,)

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(str(p), "w") as f:
            # Root attributes
            f.attrs["source_file"]  = str(SOURCE)
            f.attrs["video_fps"]    = self.video_fps
            f.attrs["num_markers"]  = N
            f.attrs["num_frames"]   = F

            ts    = f.create_group("time_series")

            # time_series/time
            ds_t = ts.create_dataset(
                "time", data=time_arr, compression="gzip", compression_opts=4)
            ds_t.attrs["units"]       = "seconds"
            ds_t.attrs["fps"]         = self.video_fps
            ds_t.attrs["description"] = "Timestamp of each video frame"

            # time_series/nodes
            nodes = ts.create_group("nodes")

            # time_series/nodes/positions
            ds_p = nodes.create_dataset(
                "positions", data=pos_arr,
                compression="gzip", compression_opts=4)
            ds_p.attrs["units"]       = "pixels"
            ds_p.attrs["axes"]        = "frame marker xy"
            ds_p.attrs["shape_note"]  = "(F, N, 2): F frames, N markers, xy=2"
            ds_p.attrs["x_axis"]      = "horizontal left-to-right"
            ds_p.attrs["y_axis"]      = "vertical top-to-bottom"
            ds_p.attrs["nan_meaning"] = "marker not detected in this frame"

            # time_series/nodes/node_ids
            ds_id = nodes.create_dataset("node_ids", data=node_ids)
            ds_id.attrs["description"] = (
                "Stable marker IDs 0..N-1 assigned by spatial sort "
                "(top-to-bottom, then left-to-right) during initialisation")

        print(f"[INFO] HDF5 saved  →  {p.resolve()}")
        print(f"       time_series/time              shape: {time_arr.shape}")
        print(f"       time_series/nodes/positions   shape: {pos_arr.shape}"
              f"  (frames={F}, markers={N}, xy=2)")
        print(f"       time_series/nodes/node_ids    shape: {node_ids.shape}")

    def print_summary(self):
        """Print per-marker detection rate to the console."""
        F = len(self._frame_ids)
        if F == 0:
            print("[WARN] No trajectory data recorded.")
            return
        print("\n" + "=" * 54)
        print("  Trajectory Summary")
        print("=" * 54)
        print(f"  Frames processed : {F}")
        print(f"  Markers          : {self.num_markers}")
        print(f"  Duration         : {self._times[-1]:.3f} s  "
              f"@ {self.video_fps:.6f} fps")
        print("-" * 54)
        print(f"  {'Marker':>8}  {'Detected':>10}  {'Lost':>8}  {'Detect %':>10}")
        for mid in range(self.num_markers):
            detected = sum(1 for row in self._pos if row[mid] is not None)
            lost     = F - detected
            pct      = 100.0 * detected / F
            print(f"  {mid:>8}  {detected:>10}  {lost:>8}  {pct:>9.1f}%")
        print("=" * 54 + "\n")


# ─────────────────────────────────────────────
#  Main tracker class
# ─────────────────────────────────────────────

class RedMarkerTracker:
    def __init__(self,
                 num_markers:     int   = 13,
                 min_area:        float = 80,
                 max_area:        float = 8000,
                 max_lost_frames: int   = 15,
                 max_match_dist:  float = 120,
                 morph_size:      int   = 3,
                 blur_kernel:     int   = 5,
                 show_ids:        bool  = True,
                 init_frames:     int   = 10):

        self.num_markers = num_markers
        self.min_area    = min_area
        self.max_area    = max_area
        self.max_lost    = max_lost_frames
        self.max_dist    = max_match_dist
        self.show_ids    = show_ids

        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_size, morph_size))
        bk = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.blur_kernel = bk

        self.trackers: List[Marker] = []

        # ── Stable spatial ID assignment ───────────────────────
        # Phase 1: accumulate init_frames of detections, cluster them,
        #          sort by position (top→bottom, left→right), assign IDs 0..N-1.
        # Phase 2: Hungarian match to fixed slots only — IDs never change.
        self._init_frames  = init_frames
        self._init_buffer: List[List[Dict]] = []
        self._ids_locked   = False
        self._anchor:      List[Tuple[float, float]] = []
        self._slots:       List[Optional[Marker]] = [None] * num_markers

        self.DOT_COLOR   = (0, 255, 0)
        self.fps_history = deque(maxlen=30)
        self._prev_time  = time.time()

    # ── Phase 1: initialisation ────────────────────────────────

    def _try_lock_ids(self, detections: List[Dict]):
        """
        Accumulate detections until init_frames frames are collected and at
        least num_markers blobs are visible in a single frame. Then sort
        mean positions top→bottom, left→right and lock permanent IDs 0..N-1.
        """
        self._init_buffer.append(detections)

        if len(self._init_buffer) < self._init_frames:
            return

        seed_frame = max(self._init_buffer, key=len)
        if len(seed_frame) < self.num_markers:
            self._init_buffer.pop(0)   # slide window, wait for more markers
            return

        seeds = sorted(seed_frame, key=lambda d: d["area"], reverse=True)[:self.num_markers]
        clusters: List[List[Tuple[float, float]]] = [[] for _ in range(self.num_markers)]

        for frame_dets in self._init_buffer:
            for det in frame_dets:
                cx, cy = det["centroid"]
                dists = [np.hypot(cx - s["centroid"][0], cy - s["centroid"][1])
                         for s in seeds]
                clusters[int(np.argmin(dists))].append((cx, cy))

        means: List[Tuple[float, float]] = []
        for k, pts in enumerate(clusters):
            if pts:
                means.append((float(np.mean([p[0] for p in pts])),
                               float(np.mean([p[1] for p in pts]))))
            else:
                means.append(seeds[k]["centroid"])

        # Sort: primary top→bottom (y), secondary left→right (x)
        order = sorted(range(self.num_markers),
                       key=lambda i: (means[i][1], means[i][0]))

        self._anchor      = [means[order[i]] for i in range(self.num_markers)]
        seed_by_order     = [seeds[order[i]] for i in range(self.num_markers)]

        for stable_id in range(self.num_markers):
            det = seed_by_order[stable_id]
            m = Marker(marker_id=stable_id,
                       centroid=det["centroid"],
                       area=det["area"],
                       bbox=det["bbox"],
                       contour=det["contour"])
            self._slots[stable_id] = m
            self.trackers.append(m)

        self._ids_locked = True
        print("[INFO] IDs locked — spatial assignment (top→bottom, left→right):")
        for i, (ax, ay) in enumerate(self._anchor):
            print(f"       M{i:02d}  anchor=({ax:.1f}, {ay:.1f})")

    # ── Phase 2: steady-state update ──────────────────────────

    def _preprocess(self, frame):
        blurred = cv2.GaussianBlur(frame, (self.blur_kernel, self.blur_kernel), 0)
        hsv  = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = build_hsv_mask(hsv, self.morph_kernel)
        return hsv, mask

    def _update_trackers(self, detections: List[Dict]):
        """
        Match detections to N fixed slots. Slots are never created or destroyed
        after init — IDs are permanently 0..NUM_MARKERS-1.
        """
        slots = [s for s in self._slots if s is not None]

        matched, unmatched_t, unmatched_d = hungarian_match(
            slots, detections, self.max_dist)

        for ti, di in matched:
            det = detections[di]
            slots[ti].area    = det["area"]
            slots[ti].bbox    = det["bbox"]
            slots[ti].contour = det["contour"]
            slots[ti].correct(det["centroid"])
            # Slowly update anchor (EMA) so it drifts with the structure's
            # resting position across a long recording
            mid = slots[ti].marker_id
            ax, ay = self._anchor[mid]
            cx, cy = det["centroid"]
            self._anchor[mid] = (ax * 0.95 + cx * 0.05, ay * 0.95 + cy * 0.05)

        for ti in unmatched_t:
            slots[ti].update_lost()
            if slots[ti].lost_frames > self.max_lost:
                slots[ti].active = False
            # Gently nudge Kalman back toward anchor when lost too long
            if slots[ti].lost_frames > self.max_lost // 2:
                mid = slots[ti].marker_id
                slots[ti].correct(self._anchor[mid])
                slots[ti].lost_frames = max(0, slots[ti].lost_frames - 1)

        # Unmatched detections are discarded — no new slots ever created

    # ── process_frame ──────────────────────────────────────────

    def process_frame(self, frame):
        """
        Process one frame.
        Returns (original, mask, overlay_on_original, overlay_on_black).

        Phase 1 (init):   buffer detections → lock spatial IDs.
        Phase 2 (locked): Hungarian match to fixed slots only.
        """
        now = time.time()
        dt  = now - self._prev_time
        self._prev_time = now
        self.fps_history.append(1.0 / dt if dt > 0 else 0)
        fps = float(np.mean(self.fps_history))

        original = frame.copy()
        _, mask  = self._preprocess(frame)
        candidates = detect_candidates(mask, self.min_area, self.max_area)

        if not self._ids_locked:
            self._try_lock_ids(candidates)
        else:
            self._update_trackers(candidates)

        h, w = frame.shape[:2]
        overlay_frame = original.copy()
        black_bg      = np.zeros((h, w, 3), dtype=np.uint8)

        GREEN = self.DOT_COLOR
        for tr in (t for t in self.trackers if t.active):
            cx, cy = int(tr.centroid[0]), int(tr.centroid[1])

            # Window 3 (overlay): green dot + contour outline + label
            cv2.circle(overlay_frame, (cx, cy), 8, GREEN, -1)
            cv2.circle(overlay_frame, (cx, cy), 8, (0, 0, 0), 1)
            cv2.drawContours(overlay_frame, [tr.contour], -1, GREEN, 1)

            # Window 4 (black bg): green dot + white ID text only
            cv2.circle(black_bg, (cx, cy), 8, GREEN, -1)
            cv2.circle(black_bg, (cx, cy), 8, (255, 255, 255), 1)

            if self.show_ids:
                label = f"M{tr.marker_id}"
                cv2.putText(overlay_frame, label, (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
                cv2.putText(overlay_frame, label, (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1)
                cv2.putText(black_bg, label, (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # HUD
        n_active = sum(1 for t in self.trackers if t.active)
        if not self._ids_locked:
            hud = (f"Initialising: {len(self._init_buffer)}/{self._init_frames}"
                   f" frames   FPS: {fps:.1f}")
        else:
            hud = f"Tracking: {n_active}/{self.num_markers}   FPS: {fps:.1f}"
        for img in (overlay_frame, black_bg):
            cv2.putText(img, hud, (18, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return original, mask, overlay_frame, black_bg


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

def main(SOURCE, HDF5_OUTPUT):
    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {SOURCE}")
        sys.exit(1)

    ret, _ = cap.read()
    if not ret:
        print("[ERROR] Cannot read first frame.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Use configured FPS; fall back to VIDEO_FPS if cap metadata is unavailable
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    video_fps    = reported_fps if reported_fps > 0 else VIDEO_FPS
    quad_size    = (QUAD_W, QUAD_H)

    tracker = RedMarkerTracker(
        num_markers     = NUM_MARKERS,
        min_area        = MIN_AREA,
        max_area        = MAX_AREA,
        max_lost_frames = MAX_LOST,
        max_match_dist  = MAX_DIST,
        morph_size      = MORPH_SIZE,
        blur_kernel     = BLUR_KERNEL,
        show_ids        = SHOW_IDS,
        init_frames     = INIT_FRAMES,
    )

    store = TrajectoryStore(num_markers=NUM_MARKERS, video_fps=video_fps)

    WINDOW = "Red Marker Tracker  [Q=quit | S=screenshot | R=reset | T=HSV tuner]"
    if SHOW_PREVIEW:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, QUAD_W, QUAD_H)

    frame_idx = 0
    print(f"[INFO] Processing : {SOURCE}")
    print(f"[INFO] Video FPS  : {video_fps:.6f} Hz")
    print("[INFO] Press Q to stop early — HDF5 is saved on exit.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[INFO] End of video after {frame_idx} frames.")
                break

            # Compute timestamp using exact FPS (avoids drift from cap timestamps)
            time_s = frame_idx / video_fps

            original, mask, overlay, black_bg = tracker.process_frame(frame)

            # Record only after IDs are locked so all stored rows are 0..N-1
            if tracker._ids_locked:
                store.record(frame_idx, time_s, tracker.trackers)

            if SHOW_PREVIEW:
                canvas = draw_quad_view(original, mask, overlay, black_bg, quad_size)
                add_labels(canvas, quad_size)
                cv2.imshow(WINDOW, canvas)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("[INFO] Quit requested by user.")
                    break
                elif key == ord('s'):
                    fname = f"screenshot_{frame_idx:05d}.png"
                    cv2.imwrite(fname, canvas)
                    print(f"[INFO] Screenshot saved: {fname}")
                elif key == ord('r'):
                    tracker.trackers.clear()
                    tracker._slots       = [None] * tracker.num_markers
                    tracker._anchor      = []
                    tracker._init_buffer = []
                    tracker._ids_locked  = False
                    print("[INFO] Tracker reset — will re-initialise.")
                elif key == ord('t'):
                    hsv_tuner(cap)

            if frame_idx % 100 == 0 and frame_idx > 0:
                n = sum(1 for t in tracker.trackers if t.active)
                print(f"  frame {frame_idx:6d}  |  t={time_s:8.4f}s  "
                      f"|  active: {n}/{NUM_MARKERS}")

            frame_idx += 1

    finally:
        cap.release()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()

        print()
        store.print_summary()

        if HDF5_OUTPUT:
            store.save_h5(SOURCE, HDF5_OUTPUT)

        print("\n[INFO] Done.")


if __name__ == "__main__":
    samples = 20
    for sample_id in range(samples):
        source = f"/Users/albertlor/Documents/Academic_PhD/origami_robotic_arm/data/soft_state_100g_near/coor_1/C1360_sample_{sample_id}.mp4"

        # Output HDF5 file
        hdf5_output = f"/Users/albertlor/Documents/Academic_PhD/origami_robotic_arm/data/soft_state_100g_near/coor_1/trajectories_sample_{sample_id}.h5"
        main(source, hdf5_output)