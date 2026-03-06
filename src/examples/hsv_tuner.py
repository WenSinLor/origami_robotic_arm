"""
Interactive HSV tuner.

Opens your video and shows the mask live while you drag sliders.
When you find values that cleanly isolate the red markers (white blobs)
without picking up the structure patches, press 'S' to print the final
values to copy into your tracker.

Controls:
    Drag sliders   — adjust HSV ranges in real time
    SPACE          — pause / unpause video
    S              — print current values to terminal
    Q              — quit
"""

import cv2
import numpy as np
from pathlib import Path

# ---- POINT THIS AT YOUR VIDEO ----
VIDEO_FILE = Path(__file__).resolve().parents[2] / "data" / "soft_state" / "coor_0" / "C1296_sample_0.mp4"
# ----------------------------------

def nothing(_):
    pass

def main():
    cap = cv2.VideoCapture(str(VIDEO_FILE))
    if not cap.isOpened():
        raise IOError(f"Cannot open: {VIDEO_FILE}")

    cv2.namedWindow("HSV Tuner", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HSV Tuner", 1400, 800)

    # Create sliders — two ranges for red (wraps around hue 0/180)
    # Range 1 (low hue end, near 0)
    cv2.createTrackbar("H1 low",  "HSV Tuner",   0, 180, nothing)
    cv2.createTrackbar("H1 high", "HSV Tuner",  10, 180, nothing)
    cv2.createTrackbar("S1 low",  "HSV Tuner", 120, 255, nothing)
    cv2.createTrackbar("S1 high", "HSV Tuner", 255, 255, nothing)
    cv2.createTrackbar("V1 low",  "HSV Tuner",  70, 255, nothing)
    cv2.createTrackbar("V1 high", "HSV Tuner", 255, 255, nothing)

    # Range 2 (high hue end, near 180)
    cv2.createTrackbar("H2 low",  "HSV Tuner", 170, 180, nothing)
    cv2.createTrackbar("H2 high", "HSV Tuner", 180, 180, nothing)
    cv2.createTrackbar("S2 low",  "HSV Tuner", 120, 255, nothing)
    cv2.createTrackbar("S2 high", "HSV Tuner", 255, 255, nothing)
    cv2.createTrackbar("V2 low",  "HSV Tuner",  70, 255, nothing)
    cv2.createTrackbar("V2 high", "HSV Tuner", 255, 255, nothing)

    # Morphology sliders
    cv2.createTrackbar("Close kernel", "HSV Tuner", 15, 40, nothing)
    cv2.createTrackbar("Min area",     "HSV Tuner", 20, 500, nothing)

    paused = False
    frame  = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        if frame is None:
            continue

        # Read slider values
        h1l  = cv2.getTrackbarPos("H1 low",  "HSV Tuner")
        h1h  = cv2.getTrackbarPos("H1 high", "HSV Tuner")
        s1l  = cv2.getTrackbarPos("S1 low",  "HSV Tuner")
        s1h  = cv2.getTrackbarPos("S1 high", "HSV Tuner")
        v1l  = cv2.getTrackbarPos("V1 low",  "HSV Tuner")
        v1h  = cv2.getTrackbarPos("V1 high", "HSV Tuner")

        h2l  = cv2.getTrackbarPos("H2 low",  "HSV Tuner")
        h2h  = cv2.getTrackbarPos("H2 high", "HSV Tuner")
        s2l  = cv2.getTrackbarPos("S2 low",  "HSV Tuner")
        s2h  = cv2.getTrackbarPos("S2 high", "HSV Tuner")
        v2l  = cv2.getTrackbarPos("V2 low",  "HSV Tuner")
        v2h  = cv2.getTrackbarPos("V2 high", "HSV Tuner")

        close_k  = max(1, cv2.getTrackbarPos("Close kernel", "HSV Tuner"))
        min_area = cv2.getTrackbarPos("Min area", "HSV Tuner")

        # Build mask
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv,
                            np.array([h1l, s1l, v1l]),
                            np.array([h1h, s1h, v1h]))
        mask2 = cv2.inRange(hsv,
                            np.array([h2l, s2l, v2l]),
                            np.array([h2h, s2h, v2h]))
        mask  = cv2.bitwise_or(mask1, mask2)

        # Morphology
        ck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        ok = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ck, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  ok, iterations=1)

        # Draw accepted contours on frame copy
        vis = frame.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        n_accepted = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            perimeter = cv2.arcLength(c, True)
            circ = (4 * np.pi * area / perimeter**2) if perimeter > 0 else 0
            colour = (0, 255, 0) if circ >= 0.5 else (0, 0, 255)
            cv2.drawContours(vis, [c], -1, colour, 2)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(vis, f"{int(area)} c={circ:.2f}",
                            (cx - 20, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            colour, 1)
            if circ >= 0.5:
                n_accepted += 1

        # Overlay info
        cv2.putText(vis, f"Accepted (green, circ>=0.5): {n_accepted}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(vis, "S=save values  SPACE=pause  Q=quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        display  = np.hstack([
            cv2.resize(vis,       (700, 400)),
            cv2.resize(mask_bgr,  (700, 400)),
        ])
        cv2.imshow("HSV Tuner", display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('s'):
            print("\n" + "="*50)
            print("Copy these into your tracker:")
            print(f"lower_red1 = np.array([{h1l}, {s1l}, {v1l}])")
            print(f"upper_red1 = np.array([{h1h}, {s1h}, {v1h}])")
            print(f"lower_red2 = np.array([{h2l}, {s2l}, {v2l}])")
            print(f"upper_red2 = np.array([{h2h}, {s2h}, {v2h}])")
            print(f"# close_kernel_size={close_k}, min_area={min_area}")
            print("="*50 + "\n")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()