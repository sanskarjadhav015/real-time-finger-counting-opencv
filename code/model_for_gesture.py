"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   model_for_gesture.py  —  Real-Time Webcam Finger Counter  (v5.0)           ║
║   OpenCV Contour Analysis  |  Convexity Defects  |  1–5 counting             ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHY THE CNN WAS REMOVED:
────────────────────────
The previous CNN-based approach failed for natural gestures because:
- It was trained on the ASL dataset, where "6" and "9" have specific hand shapes 
  that conflicted with your natural finger counts.
- It was highly sensitive to lighting and background noise.
- Preprocessing shifts (normalization/cropping) caused the model to see 
  "out-of-distribution" data, leading to false 100% confidence.

THIS NEW VERSION:
─────────────────
Uses robust geometric analysis via Convexity Defects to count gaps between fingers.
It works natively without heavy dependencies like TensorFlow.

CONTROLS:
─────────
  q / Esc  → quit
  r        → reset background (recalibrate when lighting changes)
  + / =    → increase segmentation threshold (+5)
  -        → decrease segmentation threshold (-5)
  d        → toggle debug thresholded window on/off
"""

import os
import sys
import time
import collections
import numpy as np
import cv2

# ══════════════════════════════════════════════════════════════════════════════
# 1.  TUNABLE CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# ── ROI ───────────────────────────────────────────────────────────────────────
ROI_TOP    = 100
ROI_BOTTOM = 400
ROI_LEFT   = 150
ROI_RIGHT  = 450

# ── Segmentation ──────────────────────────────────────────────────────────────
SEG_THRESHOLD    = 25          # adjustable at runtime with +/-
MIN_CONTOUR_AREA = 5000        # px² (higher to filter noise)
BG_FRAMES        = 50          # frames to learn background
ACCUM_WEIGHT     = 0.5

# ── Smoothing ─────────────────────────────────────────────────────────────────
SMOOTHING_WINDOW = 5           # rolling majority-vote for stability

# ── Display options ───────────────────────────────────────────────────────────
SHOW_DEBUG = True              # show thresholded window (toggle with d)

# ── BGR colours ───────────────────────────────────────────────────────────────
_GREEN  = (0, 220, 0)
_RED    = (0, 0, 220)
_ORANGE = (0, 165, 255)
_BLUE   = (255, 128, 0)
_WHITE  = (255, 255, 255)
_DARK   = (30, 30, 30)
_YELLOW = (0, 230, 230)
_GRAY   = (180, 180, 180)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  BACKGROUND MODEL
# ══════════════════════════════════════════════════════════════════════════════
background = None

def reset_background():
    global background
    background = None
    print("[INFO] Background model cleared — re-learning…")

def update_background(gray_frame: np.ndarray):
    global background
    if background is None:
        background = gray_frame.copy().astype("float")
        return
    cv2.accumulateWeighted(gray_frame, background, ACCUM_WEIGHT)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  HAND SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def segment_hand(gray_frame: np.ndarray, threshold: int = SEG_THRESHOLD) -> tuple | None:
    global background
    if background is None:
        return None

    # Background subtraction
    diff      = cv2.absdiff(background.astype("uint8"), gray_frame)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Morphological noise removal
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Get largest contour (presumably the hand)
    largest = max(contours, key=cv2.contourArea)
    
    # Filter tiny noise contours
    if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
        return None

    return thresh, largest


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FINGER COUNTING LOGIC (OpenCV Geometric Analysis)
# ══════════════════════════════════════════════════════════════════════════════

def get_finger_count(contour):
    """
    Calculates number of fingers based on Convexity Defects (valleys between fingers).
    
    Returns: (count, fingertips)
    - count: Integer from 1 to 5
    - fingertips: List of (x, y) coordinates for visualization
    """
    if contour is None:
        return 0, []

    # 1. Find convex hull and convexity defects
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)

    if defects is None:
        return 1, []

    # 2. Analyze each defect (valley between fingers)
    # A defect returns [start, end, far, distance]
    # start, end are indices of the points on the hull
    # far is the index of the point on the contour deepest in the valley
    
    count_defects = 0
    fingertips = []

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end   = tuple(contour[e][0])
        far   = tuple(contour[f][0])

        # Filter by triangle side lengths (robustness against tilt/noise)
        # Using the cosine rule to find the angle at 'far'
        a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        
        # Angle < 90 degrees indicates a gap between two fingers
        angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c + 1e-6)) * 180 / np.pi

        # distance check to ensure it's a significant valley
        if angle <= 90 and d > 4000:  # d is roughly distance squared (internal OpenCV unit)
            count_defects += 1
            fingertips.append(start)  # 'start' points are likely the tips

    # Correction logic:
    # If we have N gaps (defects), we have N+1 fingers visible.
    final_count = count_defects + 1
    
    # Cap at 5 fingers
    return min(max(final_count, 1), 5), fingertips


# ══════════════════════════════════════════════════════════════════════════════
# 5.  OVERLAY DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def draw_finger_results(frame, count, tips):
    """Draw count text and fingertip circles."""
    # Large label above the ROI box
    cv2.putText(frame, f"Count: {count}",
                (ROI_LEFT, ROI_TOP - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, _YELLOW, 3, cv2.LINE_AA)

    # Draw fingertip circles
    for (tx, ty) in tips:
        # Offset by ROI position
        cv2.circle(frame, (tx + ROI_LEFT, ty + ROI_TOP), 10, _RED, -1)
        cv2.circle(frame, (tx + ROI_LEFT, ty + ROI_TOP), 11, _WHITE, 2)


def draw_background_progress(frame, n_frames: int):
    bar_w  = ROI_RIGHT - ROI_LEFT
    filled = int(bar_w * (n_frames / BG_FRAMES))
    by0, by1 = ROI_BOTTOM + 12, ROI_BOTTOM + 30
    cv2.rectangle(frame, (ROI_LEFT, by0), (ROI_RIGHT, by1), _DARK, -1)
    if filled > 0:
        cv2.rectangle(frame, (ROI_LEFT, by0), (ROI_LEFT + filled, by1), _GREEN, -1)
    cv2.putText(frame, f"Learning background… {int(n_frames / BG_FRAMES * 100)}%",
                (ROI_LEFT, by1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.60, _RED, 2, cv2.LINE_AA)


def draw_fps(frame, fps: float):
    cv2.putText(frame, f"FPS {fps:.1f}", (10, 462), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _GRAY, 1)


def draw_status_strip(frame, seg_thr: int, show_debug: bool):
    flags = f"dbg={'ON' if show_debug else 'OFF'}"
    cv2.putText(frame, f"thr={seg_thr}  {flags}   r=reset  +/-=thr  d=dbg  q=quit",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (51, 255, 51), 1)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global SHOW_DEBUG

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)

    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    count_buffer  = collections.deque(maxlen=SMOOTHING_WINDOW)
    num_frames    = 0
    seg_threshold = SEG_THRESHOLD
    t_prev        = time.time()
    fps           = 0.0

    print(f"\n[INFO] Real-time finger counter running.")
    print(f"[INFO] Place hand inside the BLUE box.")
    print(f"[INFO] Keep hands OUT for the first ~{BG_FRAMES} frames (calibration).")
    print(f"[INFO] Keys: r=reset  +/-=threshold  d=debug  q/Esc=quit\n")

    while True:
        ret, frame = cam.read()
        if not ret: continue

        frame      = cv2.flip(frame, 1)      # mirror
        frame_copy = frame.copy()

        roi  = frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)

        # ── Phase 1: background learning ──────────────────────────────────────
        if num_frames < BG_FRAMES:
            update_background(gray)
            draw_background_progress(frame_copy, num_frames)
            cv2.putText(frame_copy, "Keep hands AWAY from blue box", (ROI_LEFT, ROI_BOTTOM + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, _ORANGE, 2, cv2.LINE_AA)

        # ── Phase 2: Finger Counting ──────────────────────────────────────────
        else:
            hand = segment_hand(gray, threshold=seg_threshold)

            if hand is not None:
                thresh, contour = hand

                # Draw contour on full frame
                cv2.drawContours(frame_copy, [contour + (ROI_LEFT, ROI_TOP)], -1, _GREEN, 2)

                # Get finger count
                raw_count, tips = get_finger_count(contour)
                
                # Smoothing
                count_buffer.append(raw_count)
                # Majority vote for stable count
                final_count = collections.Counter(count_buffer).most_common(1)[0][0]

                # Visuals
                draw_finger_results(frame_copy, final_count, tips)

                if SHOW_DEBUG:
                    cv2.imshow("Thresholded Hand", thresh)
            else:
                count_buffer.clear()
                cv2.putText(frame_copy, "No hand in ROI", (ROI_LEFT, ROI_TOP - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, _ORANGE, 2, cv2.LINE_AA)
                if SHOW_DEBUG:
                    blank = np.zeros((ROI_BOTTOM - ROI_TOP, ROI_RIGHT - ROI_LEFT), dtype=np.uint8)
                    cv2.imshow("Thresholded Hand", blank)

        # ── UI Elements ───────────────────────────────────────────────────────
        cv2.rectangle(frame_copy, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), _BLUE, 3)
        draw_status_strip(frame_copy, seg_threshold, SHOW_DEBUG)

        # FPS Calculation
        t_now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now
        draw_fps(frame_copy, fps)

        num_frames += 1
        cv2.imshow("Real-Time Finger Counter", frame_copy)

        # ── Keyboard Controls ─────────────────────────────────────────────────
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        elif k == ord("r"):
            reset_background()
            num_frames = 0
            count_buffer.clear()
        elif k in (ord("+"), ord("=")):
            seg_threshold = min(seg_threshold + 5, 150)
            print(f"[INFO] Threshold → {seg_threshold}")
        elif k == ord("-"):
            seg_threshold = max(seg_threshold - 5, 5)
            print(f"[INFO] Threshold → {seg_threshold}")
        elif k == ord("d"):
            SHOW_DEBUG = not SHOW_DEBUG
            if not SHOW_DEBUG: cv2.destroyWindow("Thresholded Hand")
            print(f"[INFO] Debug window → {'ON' if SHOW_DEBUG else 'OFF'}")

    cam.release()
    cv2.destroyAllWindows()
    print("[INFO] Session ended.")

if __name__ == "__main__":
    main()