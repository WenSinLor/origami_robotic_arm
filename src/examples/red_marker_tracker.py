import cv2
import numpy as np
import sys
import h5py
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

# ============================================================
# CONFIGURATION — edit these paths
# ============================================================

INPUT_VIDEO  = "/Users/albertlor/Documents/Academic_PhD/origami_robotic_arm/data/soft_state/coor_0/C1308_sample_0.mp4"   # <-- hardcoded input
OUTPUT_H5    = "/Users/albertlor/Documents/Academic_PhD/origami_robotic_arm/data/soft_state/coor_0/trajectories_sample_0.h5"  # <-- hardcoded output
DISPLAY_W    = 640
DISPLAY_H    = 360
SAVE_VIDEO   = None   # set to a path like "/path/to/out.mp4" or None

# ============================================================
# 1) Segmentation
# ============================================================

class RedSegmentationEngine:
    def __init__(self):
        self.hsv_lower1   = np.array([0,   80,  60])
        self.hsv_upper1   = np.array([12, 255, 255])
        self.hsv_lower2   = np.array([160, 80,  60])
        self.hsv_upper2   = np.array([180, 255, 255])
        self.lab_a_min    = 145
        self.lab_a_max    = 255
        self.ycrcb_cr_min = 150
        self.ycrcb_cr_max = 255

    def segment(self, frame: np.ndarray) -> np.ndarray:
        blurred   = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv       = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask_hsv1 = cv2.inRange(hsv, self.hsv_lower1, self.hsv_upper1)
        mask_hsv2 = cv2.inRange(hsv, self.hsv_lower2, self.hsv_upper2)
        mask_hsv  = cv2.bitwise_or(mask_hsv1, mask_hsv2)
        lab       = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        mask_lab  = cv2.inRange(lab[:, :, 1],
                                np.array([self.lab_a_min],    dtype=np.uint8),
                                np.array([self.lab_a_max],    dtype=np.uint8))
        ycrcb     = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb= cv2.inRange(ycrcb[:, :, 1],
                                np.array([self.ycrcb_cr_min], dtype=np.uint8),
                                np.array([self.ycrcb_cr_max], dtype=np.uint8))
        sat_mask  = (hsv[:, :, 1] > 60).astype(np.uint8) * 255
        val_mask  = (hsv[:, :, 2] > 40).astype(np.uint8) * 255
        votes     = (mask_hsv.astype(np.uint16)   // 255 +
                     mask_lab.astype(np.uint16)    // 255 +
                     mask_ycrcb.astype(np.uint16)  // 255)
        fused     = ((votes >= 2).astype(np.uint8) * 255)
        fused     = cv2.bitwise_and(fused, sat_mask)
        fused     = cv2.bitwise_and(fused, val_mask)
        return fused

    def clean_mask(self, mask: np.ndarray, min_area: int = 60) -> np.ndarray:
        ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kc = cv2.getStructuringElement(cv2.MORPH_RECT,    (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  ko, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc, iterations=2)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        clean = np.zeros_like(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean[labels == i] = 255
        return clean


# ============================================================
# 2) Detector
# ============================================================

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

class MarkerDetector:
    def __init__(self,
                 min_area: int   = 60,
                 max_area: int   = 20000,
                 min_aspect: float = 0.2,
                 max_aspect: float = 6.0,
                 solidity_thresh: float = 0.35,
                 watershed_big_blob_area: int = 2500):
        self.min_area               = min_area
        self.max_area               = max_area
        self.min_aspect             = min_aspect
        self.max_aspect             = max_aspect
        self.solidity_thresh        = solidity_thresh
        self.watershed_big_blob_area= watershed_big_blob_area

    def _watershed_segments(self, mask: np.ndarray) -> List[np.ndarray]:
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        if dist.max() <= 1e-6:
            return []
        dn = dist / (dist.max() + 1e-9)
        _, fg = cv2.threshold(dn, max(0.25, 0.45 * float(dn.max())),
                              1.0, cv2.THRESH_BINARY)
        fg  = (fg * 255).astype(np.uint8)
        bg  = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
        unk = cv2.subtract(bg, fg)
        n, markers = cv2.connectedComponents(fg)
        if n <= 1:
            return []
        markers = markers + 1
        markers[unk > 0] = 0
        cv2.watershed(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), markers)
        segs = []
        for lab in range(2, markers.max() + 1):
            seg = np.zeros_like(mask)
            seg[markers == lab] = 255
            if cv2.countNonZero(seg) >= 30:
                segs.append(seg)
        return segs

    def _validate_contour(self, cnt: np.ndarray) -> Optional[dict]:
        area = cv2.contourArea(cnt)
        if area < self.min_area or area > self.max_area:
            return None
        rect = cv2.minAreaRect(cnt)
        (_, _), (w, h), angle = rect
        if w <= 1e-3 or h <= 1e-3:
            return None
        aspect = max(w, h) / (min(w, h) + 1e-9)
        if aspect < self.min_aspect or aspect > self.max_aspect:
            return None
        hull_a   = cv2.contourArea(cv2.convexHull(cnt))
        solidity = float(area) / float(hull_a + 1e-9)
        if solidity < self.solidity_thresh:
            return None
        M = cv2.moments(cnt)
        if abs(M["m00"]) < 1e-9:
            return None
        cx_m = int(M["m10"] / M["m00"])
        cy_m = int(M["m01"] / M["m00"])
        fill  = clamp(float(area) / (w * h + 1e-9), 0.0, 1.0)
        asp_s = 1.0 - min(abs(aspect - 1.0) / 3.0, 1.0)
        conf  = clamp(0.55*solidity + 0.25*fill + 0.20*asp_s, 0.0, 1.0)
        return {"centroid": (cx_m, cy_m), "bbox": cv2.boundingRect(cnt),
                "area": float(area), "angle": float(angle),
                "contour": cnt, "confidence": float(conf), "rect": rect}

    def detect(self, mask: np.ndarray) -> List[dict]:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets: List[dict] = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            if area >= self.watershed_big_blob_area:
                x, y, w, h = cv2.boundingRect(cnt)
                roi  = mask[y:y+h, x:x+w].copy()
                segs = self._watershed_segments(roi)
                if segs:
                    for seg in segs:
                        sc, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)
                        for s in sc:
                            d = self._validate_contour(
                                s + np.array([[[x, y]]], dtype=s.dtype))
                            if d is not None:
                                dets.append(d)
                    continue
            d = self._validate_contour(cnt)
            if d is not None:
                dets.append(d)
        dets.sort(key=lambda d: d["area"], reverse=True)
        return dets


# ============================================================
# 3) Assignment
# ============================================================

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

def assign_min_cost(cost: np.ndarray,
                    max_cost: float) -> List[Tuple[int, int]]:
    if cost.size == 0:
        return []
    BIG = 1e9
    cm  = cost.astype(np.float64, copy=True)
    cm[~np.isfinite(cm)] = BIG
    if HAS_SCIPY:
        r, c = linear_sum_assignment(cm)
        return [(int(i), int(j)) for i, j in zip(r, c) if cm[i, j] <= max_cost]
    pairs, used_i, used_j = [], set(), set()
    while True:
        v = cm.min()
        if not np.isfinite(v) or v > max_cost:
            break
        i, j = np.unravel_index(cm.argmin(), cm.shape)
        if i not in used_i and j not in used_j:
            pairs.append((int(i), int(j)))
            used_i.add(i); used_j.add(j)
        cm[i, j] = BIG
    return pairs


# ============================================================
# 4) Kalman filter
# ============================================================

class SimpleKF:
    def __init__(self, x: float, y: float):
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix  = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        kf.transitionMatrix   = np.array([[1,0,1,0],[0,1,0,1],
                                          [0,0,1,0],[0,0,0,1]], np.float32)
        kf.processNoiseCov    = np.eye(4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov= np.eye(2, dtype=np.float32) * 1.0
        kf.errorCovPost       = np.eye(4, dtype=np.float32)
        kf.statePost = np.array([[x],[y],[0],[0]], np.float32)
        self.kf = kf

    def predict(self) -> Tuple[int, int]:
        p = self.kf.predict()
        return int(p[0,0]), int(p[1,0])

    def correct(self, x: float, y: float) -> Tuple[int, int]:
        c = self.kf.correct(np.array([[x],[y]], np.float32))
        return int(c[0,0]), int(c[1,0])


# ============================================================
# 5) Template Pose Tracker
# ============================================================

@dataclass
class TrackOut:
    id:          int
    centroid:    Tuple[int, int]
    observed:    bool
    lost_frames: int
    confidence:  float = 1.0

def sort_row_by_row(points_xy: np.ndarray) -> np.ndarray:
    return np.lexsort((points_xy[:, 0], points_xy[:, 1]))

def apply_affine(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.size == 0:
        return pts
    ph  = np.hstack([pts.astype(np.float32),
                     np.ones((pts.shape[0], 1), np.float32)])
    return (M @ ph.T).T.astype(np.float32)

def estimate_affine_partial(src: np.ndarray,
                            dst: np.ndarray) -> Optional[np.ndarray]:
    if src.shape[0] < 2:
        return None
    M, _ = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC,
        ransacReprojThreshold=8.0, maxIters=2000, confidence=0.99)
    return M.astype(np.float32) if M is not None else None

class TemplatePoseTracker:
    def __init__(self,
                 init_frames:      int   = 40,
                 min_init_points:  int   = 12,
                 max_ids:          Optional[int] = None,
                 max_match_dist:   float = 70.0):
        self.init_frames     = init_frames
        self.min_init_points = min_init_points
        self.max_ids         = max_ids
        self.max_match_dist  = float(max_match_dist)
        self._init_buffer: List  = []
        self.initialized         = False
        self.template_ref: Optional[np.ndarray] = None
        self.N    = 0
        self.last_M: Optional[np.ndarray] = None
        self.kfs:  Dict[int, SimpleKF] = {}
        self.lost: Dict[int, int]      = {}

    def _try_initialize(self) -> bool:
        best_pts, best_n = None, 0
        for pts, _ in self._init_buffer:
            if pts.shape[0] > best_n:
                best_n, best_pts = pts.shape[0], pts
        if best_pts is None or best_n < self.min_init_points:
            return False
        pts = best_pts.copy()
        if self.max_ids is not None:
            pts = pts[:min(len(pts), self.max_ids)]
        pts_sorted        = pts[sort_row_by_row(pts)]
        self.template_ref = pts_sorted.astype(np.float32)
        self.N            = self.template_ref.shape[0]
        self.kfs  = {i: SimpleKF(float(self.template_ref[i,0]),
                                  float(self.template_ref[i,1]))
                     for i in range(self.N)}
        self.lost = {i: 0 for i in range(self.N)}
        self.last_M       = np.array([[1,0,0],[0,1,0]], dtype=np.float32)
        self.initialized  = True
        return True

    def update(self, detections: List[dict]) -> List[TrackOut]:
        Q = (np.array([d["centroid"] for d in detections], dtype=np.float32)
             if detections else np.zeros((0, 2), np.float32))

        # ── init phase ──────────────────────────────────────
        if not self.initialized:
            if Q.shape[0] > 0:
                self._init_buffer.append((Q, detections))
            if len(self._init_buffer) >= self.init_frames:
                if not self._try_initialize():
                    self._init_buffer = self._init_buffer[len(self._init_buffer)//2:]
            # preview only
            out = []
            if Q.shape[0] > 0:
                for k, j in enumerate(sort_row_by_row(Q)):
                    out.append(TrackOut(id=k, centroid=(int(Q[j,0]), int(Q[j,1])),
                                        observed=True, lost_frames=0))
            return out

        # ── tracking phase ───────────────────────────────────
        N, M = self.N, Q.shape[0]
        P_kf   = np.array([self.kfs[i].predict() for i in range(N)], np.float32)
        P_pred = apply_affine(self.last_M, self.template_ref)
        P      = (0.7 * P_pred + 0.3 * P_kf).astype(np.float32)

        dist = (np.sqrt(np.sum((P[:, None, :] - Q[None, :, :])**2, axis=2))
                if M > 0 else np.zeros((N, 0), np.float32))
        pairs = assign_min_cost(dist, self.max_match_dist)

        # refine affine then reassign
        if len(pairs) >= 3:
            src   = np.array([self.template_ref[i] for i,_ in pairs], np.float32)
            dst   = np.array([Q[j]               for _,j in pairs], np.float32)
            M_new = estimate_affine_partial(src, dst)
            if M_new is not None:
                self.last_M = M_new
                P2    = (0.8 * apply_affine(self.last_M, self.template_ref)
                         + 0.2 * P_kf).astype(np.float32)
                dist2 = (np.sqrt(np.sum((P2[:, None, :] - Q[None, :, :])**2, axis=2))
                         if M > 0 else np.zeros((N, 0), np.float32))
                pairs = assign_min_cost(dist2, self.max_match_dist)

        map_ij = {i: j for i, j in pairs}
        out_tracks: List[TrackOut] = []
        for i in range(N):
            if i in map_ij:
                x, y = float(Q[map_ij[i], 0]), float(Q[map_ij[i], 1])
                cx, cy       = self.kfs[i].correct(x, y)
                self.lost[i] = 0
                out_tracks.append(TrackOut(id=i, centroid=(cx, cy),
                                           observed=True, lost_frames=0))
            else:
                self.lost[i] += 1
                px, py = int(P_kf[i, 0]), int(P_kf[i, 1])
                out_tracks.append(TrackOut(id=i, centroid=(px, py),
                                           observed=False,
                                           lost_frames=self.lost[i],
                                           confidence=0.0))
        out_tracks.sort(key=lambda t: t.id)
        return out_tracks


# ============================================================
# 6) HDF5 Trajectory Saver
# ============================================================

class H5TrajectorySaver:
    """
    Accumulates per-frame track data and writes one HDF5 file:

        time_series/
            time                  float64 [n_frames]       seconds
            nodes/
                positions         float32 [n_frames, n_nodes, 2]
                                  axis-2: [x, y]
                                  NaN where marker was not observed
    """

    def __init__(self, output_path: str, fps: float):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        self.output_path = output_path
        self.fps         = fps
        # frame-by-frame accumulation
        self._positions: List[Dict[int, Tuple[int, int]]] = []  # [{id:(x,y), ...}]
        self._observed:  List[Dict[int, bool]]            = []
        self._n_nodes    = 0

    def record(self, tracks: List[TrackOut]):
        """Call once per frame."""
        frame_pos = {tr.id: tr.centroid  for tr in tracks}
        frame_obs = {tr.id: tr.observed  for tr in tracks}
        self._positions.append(frame_pos)
        self._observed.append(frame_obs)
        if tracks:
            self._n_nodes = max(self._n_nodes, max(tr.id for tr in tracks) + 1)

    def save(self):
        """Write the HDF5 file."""
        n_frames = len(self._positions)
        n_nodes  = self._n_nodes

        if n_frames == 0 or n_nodes == 0:
            print("[H5] Nothing to save.")
            return

        # Build arrays
        # positions: [n_frames, n_nodes, 2]  — NaN for missing
        positions = np.full((n_frames, n_nodes, 2), np.nan, dtype=np.float32)
        for fi, (frame_pos, frame_obs) in enumerate(
                zip(self._positions, self._observed)):
            for nid, (x, y) in frame_pos.items():
                if nid < n_nodes:
                    positions[fi, nid, 0] = float(x)
                    positions[fi, nid, 1] = float(y)

        # time: [n_frames]
        time = np.arange(1, n_frames + 1, dtype=np.float64) / self.fps

        with h5py.File(self.output_path, "w") as f:
            ts = f.create_group("time_series")

            # time_series/time
            ts.create_dataset("time", data=time,
                              dtype=np.float64, compression="gzip")

            # time_series/nodes/positions
            nodes = ts.create_group("nodes")
            nodes.create_dataset("positions", data=positions,
                                 dtype=np.float32, compression="gzip")

            # useful attributes
            f.attrs["fps"]      = self.fps
            f.attrs["n_frames"] = n_frames
            f.attrs["n_nodes"]  = n_nodes
            f.attrs["axes"]     = "positions axes: [frame, node_id, (x,y)]"

        print(f"[H5] Saved: {os.path.abspath(self.output_path)}")
        print(f"     time_series/time           shape {time.shape}")
        print(f"     time_series/nodes/positions shape {positions.shape}")
        print(f"     {n_frames} frames · {n_nodes} markers")


# ============================================================
# 7) Visualization
# ============================================================

def draw_tracks(frame: np.ndarray, tracks: List[TrackOut]) -> np.ndarray:
    out = frame.copy()
    for tr in tracks:
        cx, cy = tr.centroid
        if tr.observed:
            cv2.circle(out, (cx, cy), 7, (0, 255, 0), -1)
            cv2.circle(out, (cx, cy), 7, (255, 255, 255), 1)
        else:
            cv2.circle(out, (cx, cy), 7, (0, 165, 255), 2)
            cv2.circle(out, (cx, cy), 7, (255, 255, 255), 1)
        cv2.putText(out, str(tr.id), (cx + 9, cy - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255),
                    1, cv2.LINE_AA)
    return out

def draw_centroids_black(shape: Tuple, tracks: List[TrackOut]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.uint8)
    for tr in tracks:
        cx, cy = tr.centroid
        col = (0, 255, 0) if tr.observed else (0, 165, 255)
        cv2.circle(out, (cx, cy), 7, col, -1 if tr.observed else 2)
        cv2.circle(out, (cx, cy), 7, (255, 255, 255), 1)
        cv2.putText(out, str(tr.id), (cx + 9, cy - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255),
                    2, cv2.LINE_AA)
    return out

def build_quad_view(orig, mask, overlay, black, tw, th):
    def r(img): return cv2.resize(img, (tw, th))
    def lbl(img, t):
        o = img.copy()
        cv2.putText(o, t, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 0), 2, cv2.LINE_AA)
        return o
    mb  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    top = np.hstack([lbl(r(orig),    "Original"),           lbl(r(mb),    "Red Mask")])
    bot = np.hstack([lbl(r(overlay), "Tracked (IDs)"),      lbl(r(black), "Centroids")])
    return np.vstack([top, bot])


# ============================================================
# 8) Main
# ============================================================

def main():
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {INPUT_VIDEO}")
        sys.exit(1)

    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] {INPUT_VIDEO} | {frame_w}x{frame_h} @ {fps:.2f} fps")
    print(f"[INFO] Output HDF5: {OUTPUT_H5}")
    print("[INFO] Keys: q=quit | r=reset | s=screenshot | +/-=sensitivity")

    segmenter = RedSegmentationEngine()
    detector  = MarkerDetector(min_area=10, max_area=20000)
    tracker   = TemplatePoseTracker(
        init_frames=40, min_init_points=12,
        max_ids=13,     max_match_dist=70.0)
    saver     = H5TrajectorySaver(output_path=OUTPUT_H5, fps=fps)

    writer = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(SAVE_VIDEO, fourcc, fps,
                                 (DISPLAY_W * 2, DISPLAY_H * 2))

    clahe       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    sensitivity = 0
    frame_idx   = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of stream.")
                break
            frame_idx += 1

            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            fn = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            segmenter.hsv_lower1[1] = max(50, 80 - sensitivity * 10)
            segmenter.hsv_lower2[1] = max(50, 80 - sensitivity * 10)

            raw_mask   = segmenter.segment(fn)
            clean_mask = segmenter.clean_mask(raw_mask, min_area=60)
            detections = detector.detect(clean_mask)
            tracks     = tracker.update(detections)

            saver.record(tracks)   # ← accumulate every frame

            overlay    = draw_tracks(frame, tracks)
            black_dots = draw_centroids_black(frame.shape, tracks)
            quad       = build_quad_view(frame, clean_mask, overlay,
                                         black_dots, DISPLAY_W, DISPLAY_H)

            cv2.imshow("Red Marker Tracker [q/r/s/+/-]", quad)
            if writer:
                writer.write(quad)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                tracker = TemplatePoseTracker(
                    init_frames=40, min_init_points=12,
                    max_ids=13,     max_match_dist=70.0)
                print("[INFO] Tracker reset.")
            elif key == ord('s'):
                fname = f"screenshot_{frame_idx:06d}.png"
                cv2.imwrite(fname, quad)
                print(f"[INFO] Screenshot saved: {fname}")
            elif key in (ord('+'), ord('=')):
                sensitivity = min(sensitivity + 1, 5)
                print(f"[INFO] Sensitivity: {sensitivity}")
            elif key == ord('-'):
                sensitivity = max(sensitivity - 1, -3)
                print(f"[INFO] Sensitivity: {sensitivity}")

    finally:
        saver.save()    # ← write HDF5 on exit (even on Ctrl+C)
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()