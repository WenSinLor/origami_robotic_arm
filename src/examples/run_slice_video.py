import cv2
import os

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

INPUT_PATH  = "/Users/albertlor/Documents/Academic_PhD/origami_robotic_arm/data/stiff_state_100g/coor_0/C1381.MP4"   # Path to the source video
OUTPUT_DIR  = "/Users/albertlor/Documents/Academic_PhD/origami_robotic_arm/data/stiff_state_100g/coor_0/"     # Directory where slices will be saved

START_TIME     = 5.5    # Start time of the first slice (seconds)
SLICE_DURATION = 7    # Duration of each slice (seconds)
INTERVAL       = 8   # Time between the start of each slice (seconds)
NUM_SLICES     = 20    # Total number of slices to extract

# ──────────────────────────────────────────────────────────────────────────────


def slice_video():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {INPUT_PATH}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    base_name = os.path.splitext(os.path.basename(INPUT_PATH))[0]

    print(f"Input:          {INPUT_PATH}")
    print(f"Resolution:     {width}x{height}  |  FPS: {fps:.3f}")
    print(f"Output dir:     {OUTPUT_DIR}")
    print(f"Start time:     {START_TIME}s")
    print(f"Slice duration: {SLICE_DURATION}s")
    print(f"Interval:       {INTERVAL}s")
    print(f"Num slices:     {NUM_SLICES}")
    print("-" * 48)

    frames_per_slice = int(round(fps * SLICE_DURATION))

    for i in range(NUM_SLICES):
        slice_start_sec = START_TIME + i * INTERVAL
        start_frame     = int(round(slice_start_sec * fps))
        output_path     = os.path.join(OUTPUT_DIR, f"{base_name}_sample_{i}.MP4")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Slice {i + 1}/{NUM_SLICES} — start: {slice_start_sec}s (frame {start_frame})  →  {os.path.basename(output_path)}")

        for f in range(frames_per_slice):
            ret, frame = cap.read()
            if not ret:
                print(f"  WARNING: Video ended early at frame {f + 1}/{frames_per_slice}")
                break
            out.write(frame)

        out.release()
        print(f"  Done.")

    cap.release()
    print("-" * 48)
    print("All slices complete.")


if __name__ == "__main__":
    slice_video()