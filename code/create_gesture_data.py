"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   create_gesture_data.py  —  Dataset Collection Tool  (FINAL v4)           ║
║   Sign Language Recognition  |  Numbers 1-10  (extensible)                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

PURPOSE
───────
  Collects your OWN training/test images from webcam.
  Use this when you want to train on YOUR hand shape instead of Kaggle data.
  After collection, set LOCAL_DATASET_ROOT in DataFlair_trainCNN.py to point
  here, and run training without Kaggle download.

WHAT THIS SCRIPT SAVES (and why it matches inference exactly)
─────────────────────────────────────────────────────────────
  1. Background-subtracted binary image (same as inference)
  2. Morphological clean-up: MORPH_OPEN + MORPH_CLOSE (same kernel as inference)
  3. Tight bounding-rect crop with CROP_PAD (same as inference TIGHT_CROP mode)
  4. Saved as grayscale JPEG (loaded as RGB by ImageDataGenerator — R=G=B)
  5. Folder structure:  dataset/train/<gesture_number>/
                        dataset/test/<gesture_number>/

HOW TO USE
──────────
  1. python create_gesture_data.py
  2. Keep hand OUT of blue box for first 60 frames (background learning).
  3. Show gesture inside blue box.  Watch the "Preview" window — that exact
     image is what the model will see.  Adjust if it looks noisy.
  4. Script auto-saves up to IMAGES_PER_CLASS images, then waits for + key.
  5. Press + to advance to next gesture number (1 → 2 → … → 10).
  6. Repeat for test split by pressing  s.

KEYBOARD SHORTCUTS
──────────────────
  + / =   →  next gesture number
  -       →  previous gesture number
  s       →  toggle train / test split
  r       →  reset background (re-calibrate after lighting change)
  q / Esc →  quit

IMPORTANT — MUST MATCH model_for_gesture.py
────────────────────────────────────────────
  The following constants MUST be identical in both scripts:
    SEG_THRESHOLD, CROP_PAD, MIN_CONTOUR_AREA
    GaussianBlur kernel (9,9)
    MORPH_OPEN / MORPH_CLOSE kernel size and iterations
  Any mismatch = training/inference domain gap = wrong predictions.
"""

import cv2
import numpy as np
import os

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  — keep CROP_PAD and morphology params in sync with model_for_gesture.py
# ══════════════════════════════════════════════════════════════════════════════
IMAGES_PER_CLASS = 300
DATASET_ROOT     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "dataset")
AUTO_ADVANCE     = False          # True = auto-jump when class is complete

# Gesture labels (folder names are the digit strings)
_GESTURE_NAMES = {
    1: "One",  2: "Two",   3: "Three", 4: "Four",  5: "Five",
    6: "Six",  7: "Seven", 8: "Eight", 9: "Nine",  10: "Ten",
}

# ── ROI (MUST match model_for_gesture.py) ────────────────────────────────────
ROI_TOP    = 100
ROI_BOTTOM = 400
ROI_LEFT   = 150
ROI_RIGHT  = 450

# ── Segmentation (MUST match model_for_gesture.py) ───────────────────────────
BG_FRAMES        = 60
ACCUM_WEIGHT     = 0.5
SEG_THRESHOLD    = 25
MIN_CONTOUR_AREA = 3000       # px² — same as inference MIN_CONTOUR_AREA

# ── Tight crop (MUST match model_for_gesture.py TIGHT_CROP padding) ──────────
CROP_PAD = 10

# BGR draw colours
_BLUE   = (255, 128, 0)
_GREEN  = (0, 220, 0)
_RED    = (0, 0, 220)
_ORANGE = (0, 165, 255)
_YELLOW = (0, 230, 230)
_WHITE  = (255, 255, 255)
_DARK   = (30, 30, 30)


# ══════════════════════════════════════════════════════════════════════════════
# BACKGROUND MODEL
# ══════════════════════════════════════════════════════════════════════════════
background = None


def reset_background():
    global background
    background = None


def cal_accum_avg(frame: np.ndarray):
    global background
    if background is None:
        background = frame.copy().astype("float")
        return
    cv2.accumulateWeighted(frame, background, ACCUM_WEIGHT)


# ══════════════════════════════════════════════════════════════════════════════
# HAND SEGMENTATION  ← MUST be byte-for-byte identical to model_for_gesture.py
# ══════════════════════════════════════════════════════════════════════════════

def segment_hand(frame: np.ndarray) -> tuple | None:
    """
    Returns (thresh_img, largest_contour) or None.

    Pipeline (must match inference):
      absdiff → threshold → MORPH_OPEN(3×3, 1 iter) → MORPH_CLOSE(3×3, 1 iter)
      → largest contour above MIN_CONTOUR_AREA
    """
    global background
    if background is None:
        return None

    diff      = cv2.absdiff(background.astype("uint8"), frame)
    _, thresh = cv2.threshold(diff, SEG_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Morphological noise removal — MUST match model_for_gesture.py
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh.copy(),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
        return None

    return thresh, largest


# ══════════════════════════════════════════════════════════════════════════════
# TIGHT CROP  ← MUST be identical to preprocess_for_model() in model_for_gesture.py
# ══════════════════════════════════════════════════════════════════════════════

def get_tight_crop(thresh: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """
    Crop the bounding rect of the hand contour + CROP_PAD pixels on each side.
    This is the exact image the model sees at inference time (TIGHT_CROP=True).
    """
    x, y, w, h = cv2.boundingRect(contour)
    H, W       = thresh.shape[:2]
    x1 = max(0, x - CROP_PAD)
    y1 = max(0, y - CROP_PAD)
    x2 = min(W, x + w + CROP_PAD)
    y2 = min(H, y + h + CROP_PAD)
    crop = thresh[y1:y2, x1:x2]
    return crop if crop.size > 0 else thresh   # fallback: full ROI


# ══════════════════════════════════════════════════════════════════════════════
# OVERLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def draw_progress_bar(frame, count: int, total: int,
                      x: int, y: int, w: int, h: int = 20):
    fill = int(w * min(count / max(total, 1), 1.0))
    cv2.rectangle(frame, (x, y), (x + w, y + h), _DARK, -1)
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + h), _GREEN, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), _WHITE, 1)
    cv2.putText(frame, f"{count}/{total}",
                (x + w + 8, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, _WHITE, 1)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError(
            "[ERROR] Cannot open webcam (index 0).  "
            "Try cv2.VideoCapture(1) in the code."
        )

    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    split      = "train"
    element    = 1
    num_frames = 0
    num_imgs   = 0

    print(f"\n[INFO] Dataset root: {DATASET_ROOT}")
    print(f"[INFO] Collecting '{split}' data  |  gesture: {element} "
          f"({_GESTURE_NAMES[element]})")
    print("[INFO] Keys:  +/-=gesture   s=toggle split   r=reset bg   q/Esc=quit\n")

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        frame      = cv2.flip(frame, 1)
        frame_copy = frame.copy()

        roi        = frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]
        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

        # ── Phase 1: background learning ──────────────────────────────────────
        if num_frames < BG_FRAMES:
            cal_accum_avg(gray_frame)
            pct = int(num_frames / BG_FRAMES * 100)
            cv2.putText(frame_copy,
                        f"Learning background… {pct}%  — keep hands OUT of box",
                        (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.65, _RED, 2)
            draw_progress_bar(frame_copy, num_frames, BG_FRAMES,
                              ROI_LEFT, ROI_BOTTOM + 15,
                              ROI_RIGHT - ROI_LEFT)

        # ── Phase 2: preview (no saving yet) ─────────────────────────────────
        elif num_frames < BG_FRAMES + 40:
            hand = segment_hand(gray_frame)
            cv2.putText(frame_copy,
                        f"Show gesture [{element}: {_GESTURE_NAMES[element]}]  — adjust",
                        (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.70, _GREEN, 2)
            if hand:
                thresh, seg = hand
                crop = get_tight_crop(thresh, seg)
                cv2.drawContours(frame_copy,
                                 [seg + (ROI_LEFT, ROI_TOP)], -1, _GREEN, 1)
                cv2.imshow("Preview — what model will see", crop)

        # ── Phase 3: capture ──────────────────────────────────────────────────
        else:
            hand = segment_hand(gray_frame)

            if hand and num_imgs < IMAGES_PER_CLASS:
                thresh, seg = hand
                # Tight crop — SAME as inference preprocessing
                crop = get_tight_crop(thresh, seg)

                cv2.drawContours(frame_copy,
                                 [seg + (ROI_LEFT, ROI_TOP)], -1, _GREEN, 2)
                cv2.imshow("Preview — what model will see", crop)

                # Save as grayscale (already single-channel after threshold)
                save_dir  = os.path.join(DATASET_ROOT, split, str(element))
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{num_imgs}.jpg")
                cv2.imwrite(save_path, crop)
                num_imgs += 1

                if AUTO_ADVANCE and num_imgs >= IMAGES_PER_CLASS:
                    print(f"\a[INFO] Class {element} done!  Auto-advancing…")
                    element    = min(element + 1, 10)
                    num_imgs   = 0
                    num_frames = 0
                    reset_background()
                    print(f"[INFO] → gesture: {element} ({_GESTURE_NAMES[element]})")

            elif not hand:
                cv2.putText(frame_copy,
                            "No hand detected — adjust position",
                            (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.70, _ORANGE, 2)

            # ── Status bar ────────────────────────────────────────────────────
            done         = num_imgs >= IMAGES_PER_CLASS
            status_color = _GREEN if done else _YELLOW
            status_text  = (f"[{split.upper()}]  "
                            f"{_GESTURE_NAMES[element]} ({element})  ")
            status_text += "DONE — press + for next" if done else f"{num_imgs}/{IMAGES_PER_CLASS}"
            cv2.putText(frame_copy, status_text,
                        (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.70, status_color, 2)
            draw_progress_bar(frame_copy, num_imgs, IMAGES_PER_CLASS,
                              ROI_LEFT, ROI_BOTTOM + 15,
                              ROI_RIGHT - ROI_LEFT)

        # ── ROI box ───────────────────────────────────────────────────────────
        cv2.rectangle(frame_copy,
                      (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), _BLUE, 3)
        cv2.putText(frame_copy,
                    "Sign Language | r=reset  s=split  +/-=gesture  q=quit",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (51, 255, 51), 1)

        num_frames += 1
        cv2.imshow("Sign Detection — Data Collection", frame_copy)

        k = cv2.waitKey(1) & 0xFF

        if k in (27, ord("q")):
            break
        elif k in (ord("+"), ord("=")):
            element    = min(element + 1, 10)
            num_imgs   = 0
            num_frames = 0
            reset_background()
            print(f"[INFO] → gesture: {element} ({_GESTURE_NAMES[element]})")
        elif k == ord("-"):
            element    = max(element - 1, 1)
            num_imgs   = 0
            num_frames = 0
            reset_background()
            print(f"[INFO] → gesture: {element} ({_GESTURE_NAMES[element]})")
        elif k == ord("s"):
            split      = "test" if split == "train" else "train"
            num_imgs   = 0
            num_frames = 0
            reset_background()
            print(f"[INFO] → split: {split}")
        elif k == ord("r"):
            reset_background()
            num_frames = 0
            print("[INFO] Background reset.")

    cam.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Data collection finished.")
    print(f"[INFO] Dataset saved under: {os.path.abspath(DATASET_ROOT)}")
    print(f"\nNext step:\n  Set LOCAL_DATASET_ROOT = r\"{os.path.abspath(DATASET_ROOT)}\" "
          f"in DataFlair_trainCNN.py\n  then run:  python DataFlair_trainCNN.py\n")


if __name__ == "__main__":
    main()