import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing.tracking import MarkerTracker, VideoLoader


def main():
    # --- MARKER CONFIGURATION ---
    NUM_MARKERS = 37
    # ----------------------------

    # 1. Setup Paths
    current_script_dir = Path(__file__).parent.resolve()
    DATA_DIR = current_script_dir.parent.parent / "data"
    VIDEO_FILE = DATA_DIR / "soft_state" / "coor_0" / "C1296_sample_0.MP4"
    OUTPUT_FILE = DATA_DIR / "experiment_data" / "coor_0" / "spring_mass_data.npz"

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 2. Setup Tracker
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    tracker = MarkerTracker(lower_red1, upper_red1, lower_red2, upper_red2, min_area=100, max_area=20000)

    # 3. Setup Visualization
    window_name = "2x2 View: Clean | Mask | Tracking | Dots"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # --- DATA STORAGE ---
    raw_trajectory_data = []
    valid_frames_count = 0

    print(f"Processing: {VIDEO_FILE}")
    print(f"Expecting {NUM_MARKERS} markers.")

    with VideoLoader(str(VIDEO_FILE)) as loader:
        total_frames = int(loader.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video Length: {total_frames} frames")

        for frame_idx, frame in enumerate(loader.stream_frames()):
            clean_frame = frame.copy()
            mask, centroids = tracker.process_frame(clean_frame, frame_idx)
            sorted_centroids = []
            
            # Sort top-to-bottom, then left-to-right as tiebreak
            sorted_centroids = sorted(centroids, key=lambda p: (p[1], p[0]))

            frame_block = [[cx, cy, 0] for (cx, cy) in sorted_centroids]
            raw_trajectory_data.append(frame_block)
            valid_frames_count += 1

            if len(centroids) != NUM_MARKERS:
                if len(centroids) > 0:
                    print(f"Frame {frame_idx}: Found {len(centroids)} markers (expected {NUM_MARKERS}).")

            # --- VISUALIZATION ---
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            frame_with_dots = frame.copy()
            black_with_dots = np.zeros_like(clean_frame)

            for i, centroid in enumerate(sorted_centroids):
                cv2.circle(frame_with_dots, centroid, 15, (0, 255, 0), -1)
                cv2.circle(black_with_dots, centroid, 15, (0, 255, 0), -1)

                text_pos = (centroid[0] - 15, centroid[1] - 20)
                cv2.putText(black_with_dots, str(i), text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 0), 4)   # black outline
                cv2.putText(black_with_dots, str(i), text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2)  # white text
                cv2.putText(frame_with_dots, str(i), (centroid[0] - 10, centroid[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            row1 = np.hstack([clean_frame, mask_bgr])
            row2 = np.hstack([frame_with_dots, black_with_dots])
            grid = np.vstack([row1, row2])

            cv2.imshow(window_name, grid)
            if len(centroids) != NUM_MARKERS:
                key = cv2.waitKey(0) & 0xFF  # pause until keypress
            else:
                key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cv2.destroyAllWindows()

    # --- SAVE ---
    final_array = np.array(raw_trajectory_data)

    print("-" * 30)
    print("Processing Complete.")
    print(f"Total Valid Frames: {valid_frames_count}")
    print(f"Final Data Shape: {final_array.shape}")

    if final_array.size > 0:
        np.savez_compressed(OUTPUT_FILE, trajectories=final_array)
        print(f"Saved compressed data to {OUTPUT_FILE}")
    else:
        print("No valid frames were processed. Nothing to save.")


if __name__ == "__main__":
    main()
