import cv2
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


class MarkerTracker:
    """
    Identifies red markers, cleans noise, and calculates centroids for all found markers.

    Key fix: adds morphological CLOSING before opening to merge fragments of a single
    marker blob that got split by the HSV mask (e.g. due to specular highlights or
    non-uniform lighting). Then merges any centroids that are suspiciously close
    together (closer than merge_dist pixels) into a single centroid.
    """

    def __init__(self, lower_hsv1, upper_hsv1, lower_hsv2, upper_hsv2,
                 min_area=10000, max_area=20000,
                 close_kernel_size=15,
                 merge_dist=20):
        """
        Args:
            lower_hsv1, upper_hsv1: First HSV range for red.
            lower_hsv2, upper_hsv2: Second HSV range for red (wraps around 180).
            min_area: Minimum contour area to be considered a marker.
            max_area: Maximum contour area to be considered a marker.
            close_kernel_size: Size of the elliptical kernel used for morphological
                closing. Increase this if a single physical marker is still being
                detected as multiple fragments. Rule of thumb: set to ~50-75% of
                the marker's pixel diameter.
            merge_dist: If two detected centroids are closer than this many pixels,
                they are assumed to be fragments of the same physical marker and
                merged into their average position. Set to ~the marker diameter.
        """
        self.min_area = min_area
        self.max_area = max_area
        self.lower_hsv1 = lower_hsv1
        self.upper_hsv1 = upper_hsv1
        self.lower_hsv2 = lower_hsv2
        self.upper_hsv2 = upper_hsv2

        # Small kernel for opening — removes isolated noise speckles
        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # Larger elliptical kernel for closing — fills internal holes/gaps in a
        # marker blob caused by specular highlights or non-uniform illumination
        self.close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))

        # Distance threshold for merging nearby centroids (same physical marker)
        self.merge_dist = merge_dist

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        # 1. HSV Thresholding with two ranges
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_hsv1, self.upper_hsv1)
        mask2 = cv2.inRange(hsv, self.lower_hsv2, self.upper_hsv2)
        mask = cv2.bitwise_or(mask1, mask2)

        # 2. CLOSE first — merges fragments within the same marker blob.
        #    Use a larger kernel here; if a marker has a bright specular spot in
        #    the middle the close will bridge the gap before we find contours.
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel, iterations=2)

        # 3. OPEN after closing — removes any small noise blobs that closing
        #    may have created or left behind.
        mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, self.open_kernel, iterations=1)

        # 4. Find centroids
        centroids = self._find_marker_centroids(mask_clean, frame_idx)

        # 5. Merge centroids that are too close to be separate physical markers
        centroids = self._merge_nearby_centroids(centroids)

        return mask_clean, centroids

    def _find_marker_centroids(self, mask: np.ndarray, frame_idx: int) -> List[Tuple[int, int]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return []

        contour_areas = [cv2.contourArea(c) for c in contours]
        print(f"Frame {frame_idx}: {[round(a,1) for a in contour_areas]}")

        found_centroids = []
        for c, area in zip(contours, contour_areas):
            if area < self.min_area or area > self.max_area:
                continue

            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                found_centroids.append((cX, cY))

        return found_centroids

    def _merge_nearby_centroids(self, centroids: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Merge clusters of centroids that are closer than self.merge_dist pixels.

        Uses a simple union-find / single-linkage approach:
          - Build a graph where two centroids are connected if their distance < merge_dist
          - Find connected components
          - Replace each component with its centroid (mean position)

        This handles the case where closing wasn't enough to fully merge a split
        marker blob, so it still produced 2-3 nearby contours.
        """
        if len(centroids) <= 1:
            return centroids

        pts = np.array(centroids, dtype=float)
        n   = len(pts)

        # Union-Find
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            parent[find(a)] = find(b)

        # Connect pairs closer than merge_dist
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(pts[i] - pts[j]) < self.merge_dist:
                    union(i, j)

        # Collect components and compute mean position
        from collections import defaultdict
        groups = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)

        merged = []
        for members in groups.values():
            mean_pt = pts[members].mean(axis=0)
            merged.append((int(mean_pt[0]), int(mean_pt[1])))

        return merged


class VideoLoader:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file: {self.video_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()

    def stream_frames(self):
        """Generator that yields frames one by one, ensuring it stops at EOF."""
        if not self.cap.isOpened():
            return

        total_frames  = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0

        while self.cap.isOpened():
            if current_frame >= total_frames:
                break
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame
            current_frame += 1


class VideoProcessor:
    def __init__(self, video_path, xml_path, tracker):
        self.video_path = Path(video_path)
        self.xml_path   = Path(xml_path)
        self.tracker    = tracker
        self.fps        = 29.97
        self.start_timestamp = 0.0

    def get_start_timestamp(self):
        try:
            tree = ET.parse(self.xml_path)
            for elem in tree.getroot().iter():
                if 'CreationDate' in elem.tag:
                    dt = datetime.fromisoformat(elem.attrib['value'])
                    return dt.timestamp()
        except Exception as e:
            print(f"Warning: XML Timestamp error ({e}). Defaulting to 0.0")
        return 0.0

    def process_video(self, grid_rows: int, grid_cols: int, visualize: bool = False):
        print(f"Processing Video: {self.video_path.name}")
        self.start_timestamp = self.get_start_timestamp()

        expected_markers = grid_rows * grid_cols
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")

        raw_trajectories, frame_indices = [], []
        frame_count  = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        window_name = f"2x2 View: {self.video_path.name}"
        if visualize:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                clean_frame     = frame.copy()
                frame_with_dots = frame.copy()
                mask, centroids = self.tracker.process_frame(clean_frame, frame_count)

                # Sort: top-to-bottom row by row, left-to-right within each row
                y_sorted = sorted(centroids, key=lambda p: p[1])
                sorted_centroids = []
                for i in range(grid_rows):
                    row = y_sorted[i * grid_cols:(i + 1) * grid_cols]
                    sorted_centroids.extend(sorted(row, key=lambda p: p[0]))

                block = [[c[0], c[1], 0] for c in sorted_centroids]
                raw_trajectories.append(block)
                frame_indices.append(frame_count)

                if len(centroids) != expected_markers and len(centroids) > 0:
                    print(f"Frame {frame_count}: Found {len(centroids)} markers "
                          f"(expected {expected_markers}).")

                if visualize:
                    mask_bgr        = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    black_with_dots = np.zeros_like(clean_frame)

                    for i, centroid in enumerate(sorted_centroids):
                        cv2.circle(frame_with_dots, centroid, 15, (0, 255, 0), -1)
                        cv2.circle(black_with_dots, centroid, 15, (0, 255, 0), -1)
                        text_pos = (centroid[0] - 15, centroid[1] - 20)
                        cv2.putText(black_with_dots, str(i), text_pos,
                                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 4)
                        cv2.putText(black_with_dots, str(i), text_pos,
                                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
                        cv2.putText(frame_with_dots, str(i),
                                    (centroid[0] - 10, centroid[1] - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    grid = np.vstack([np.hstack([clean_frame, mask_bgr]),
                                      np.hstack([frame_with_dots, black_with_dots])])
                    cv2.imshow(window_name, grid)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if frame_count % 100 == 0:
                    print(f"  Frame {frame_count}/{total_frames}...", end='\r')
                frame_count += 1

        finally:
            cap.release()
            if visualize:
                cv2.destroyAllWindows()

        print(f"\nTracking Complete. Valid Frames: {len(raw_trajectories)}")
        return np.array(raw_trajectories), np.array(frame_indices)