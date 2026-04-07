"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   DataFlair_trainCNN.py  —  Training Script  (FINAL v4)                    ║
║   Sign Language Recognition  |  Numbers 1-10                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

ROOT-CAUSE ANALYSIS — why the original gave 100 % false confidence + wrong labels
──────────────────────────────────────────────────────────────────────────────────
PROBLEM 1 — PREPROCESSING MISMATCH  ← THE #1 KILLER
  Original training used:
      ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)
  vgg16.preprocess_input does:
      • Converts RGB→BGR
      • Subtracts ImageNet channel means  [103.939, 116.779, 123.68]
      • Leaves values roughly in range [-128, +151]
  Inference (model_for_gesture.py) did:
      thresholded = np.reshape(thresholded, (1,64,64,3))
      pred = model.predict(thresholded)  ← raw uint8 [0,255], no normalisation at all
  Result: model always sees completely out-of-distribution values → one class
  dominates regardless of gesture → 100 % false confidence is guaranteed.

  FIX: Use rescale=1./255 in ImageDataGenerator.
       Inference also divides by 255.0.  Both pipelines now see identical [0,1] floats.

PROBLEM 2 — LABEL ORDERING never written to disk
  ImageDataGenerator.flow_from_directory uses alphabetical sort of FOLDER NAMES.
  Folder names are "1","10","2","3","4","5","6","7","8","9".
  Alphabetical order: "1" < "10" < "2" < "3" … < "9"
  So class index 0 → folder "1" (One), index 1 → folder "10" (Ten),
  index 2 → folder "2" (Two), …, index 9 → folder "9" (Nine).
  The original word_dict = {0:'One',1:'Ten',2:'Two',…} was CORRECT for this order,
  but it was never saved — any future run had to know this by heart.
  FIX: Save label_map.json automatically at training time so inference always
  loads the ground-truth mapping.

PROBLEM 3 — MORPHOLOGICAL CLEAN-UP missing during training data collection
  The original create_gesture_data.py saved the raw thresholded image with no
  morphological clean-up, producing noisy images.  Inference applied MORPH_OPEN+
  MORPH_CLOSE.  FIX: both scripts now apply the SAME morphological pipeline.

PROBLEM 4 — FULL ROI vs TIGHT CROP inconsistency
  Training saved the full 300×300 ROI.  Inference passed only the contour crop.
  FIX: both scripts save/use a tight bounding-rect crop with 10-pixel padding.

PROBLEM 5 — model.compile() called TWICE with different optimisers
  The original file compiled with Adam, then immediately re-compiled with SGD —
  the Adam weights were discarded.  This is harmless but confusing; cleaned up.

PROBLEM 6 — No train/val split in original DataFlair code
  Used the test folder as validation during training.  Fine for a toy project,
  kept as-is but clarified in comments.

HOW TO USE
──────────
  1. Install: pip install tensorflow opencv-python kagglehub matplotlib
  2. Set up Kaggle API key (see below)
  3. python DataFlair_trainCNN.py
  4. Model saved as  best_model_dataflair3.h5  and  label_map.json
     (in the same folder as this script)

KAGGLE API KEY SETUP (one-time)
────────────────────────────────
  Option A (recommended):
      Place kaggle.json at  ~/.kaggle/kaggle.json   (Linux/Mac)
                         or  C:\\Users\\<you>\\.kaggle\\kaggle.json  (Windows)
  Option B (environment variables):
      set KAGGLE_USERNAME=your_username
      set KAGGLE_KEY=your_api_key
  Get your key from: https://www.kaggle.com/settings → API → Create New Token
"""

# ══════════════════════════════════════════════════════════════════════════════
# 0.  IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import os
import json
import shutil
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.simplefilter(action="ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPool2D, Flatten,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (ReduceLROnPlateau, EarlyStopping,
                                        ModelCheckpoint)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIG  (change only these values)
# ══════════════════════════════════════════════════════════════════════════════
IMG_SIZE        = 64          # px — must match create_gesture_data.py and inference
BATCH_SIZE      = 32
EPOCHS          = 20
LEARNING_RATE   = 0.001
DROPOUT_RATE    = 0.35
NUM_CLASSES     = 10

_HERE           = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(_HERE, "best_model_dataflair3.h5")
LABEL_MAP_PATH  = os.path.join(_HERE, "label_map.json")

# Kaggle dataset slug — change if you use a different dataset
KAGGLE_DATASET  = "grassknoted/asl-alphabet"   # fallback slug; override below
# For the DataFlair / sign-language-mnist style with folders 1-10 use:
KAGGLE_DATASET = "pranavsharma1670/sign-language-recognition-dataset"

# Override KAGGLE_DATASET here if you collected your own data with
# create_gesture_data.py and want to skip the Kaggle download:
LOCAL_DATASET_ROOT = None   # e.g. r"C:\myproject\dataset"
#   Set to the folder that contains  train/  and  test/  sub-directories.
#   When set, the Kaggle download is skipped entirely.

# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATASET ACQUISITION
# ══════════════════════════════════════════════════════════════════════════════

def get_dataset_paths() -> tuple[str, str]:
    """
    Returns (train_path, test_path).

    Priority:
      1. LOCAL_DATASET_ROOT  — user-collected data from create_gesture_data.py
      2. kagglehub cache     — automatic download via kagglehub API
    """
    # ── Option 1: local dataset (create_gesture_data.py output) ──────────────
    if LOCAL_DATASET_ROOT is not None:
        train_path = os.path.join(LOCAL_DATASET_ROOT, "train")
        test_path  = os.path.join(LOCAL_DATASET_ROOT, "test")
        if not os.path.isdir(train_path):
            raise FileNotFoundError(
                f"[ERROR] train/ not found inside LOCAL_DATASET_ROOT:\n"
                f"  {LOCAL_DATASET_ROOT}\n"
                f"  Run create_gesture_data.py first."
            )
        print(f"[INFO] Using local dataset: {LOCAL_DATASET_ROOT}")
        return train_path, test_path

    # ── Option 2: Kaggle download ─────────────────────────────────────────────
    try:
        import kagglehub
    except ImportError:
        raise ImportError(
            "[ERROR] kagglehub not installed.\n"
            "  pip install kagglehub\n"
            "  Then make sure your Kaggle API key is configured."
        )

    print(f"[INFO] Downloading / loading from Kaggle cache: {KAGGLE_DATASET}")
    dataset_root = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"[INFO] Kaggle cache path: {dataset_root}")

    # Walk the download to find train/ and test/ (varies by dataset)
    # FIX for this dataset structure
    dataset_folder = os.path.join(dataset_root, "Dataset_1_12")

    train_path = dataset_folder
    test_path = dataset_folder   # we will use same folder (no test folder)

    print(f"[INFO] Using dataset folder: {train_path}")

    if train_path is None:
        # Some Kaggle datasets use different structures — list what we found
        dirs = []
        for root, subdirs, _ in os.walk(dataset_root):
            for d in subdirs:
                dirs.append(os.path.join(root, d))
        print("[WARN] Could not auto-detect train/ folder.  Found directories:")
        for d in dirs[:20]:
            print(f"  {d}")
        raise RuntimeError(
            "[ERROR] Set LOCAL_DATASET_ROOT manually or pick a different KAGGLE_DATASET."
        )

    print(f"[INFO] train path: {train_path}")
    print(f"[INFO] test  path: {test_path}")
    return train_path, test_path


def _find_subdir(root: str, name: str) -> str | None:
    """Recursively search for a sub-directory called `name`."""
    for dirpath, dirnames, _ in os.walk(root):
        if name in dirnames:
            return os.path.join(dirpath, name)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DATA GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def build_generators(train_path: str, test_path: str):
    """
    FIX 1: rescale=1./255  — the ONLY correct normalisation for this pipeline.
           DO NOT use vgg16.preprocess_input here.  vgg16.preprocess_input
           subtracts ImageNet means and outputs values in [-128, +151].
           Inference divides by 255 → completely different range → wrong predictions.
           rescale=1/255 → both training and inference see values in [0, 1].

    Augmentation (train only):
      - Small rotation / width-height shifts keep the model robust to hand position.
      - Horizontal flip is DISABLED: handshape for "3" mirrored ≠ "3".
      - No aggressive augmentation — the images are binary thresholds, not photos.
    """
    train_gen = ImageDataGenerator(
        rescale          = 1.0 / 255,      # FIX 1
        rotation_range   = 10,
        width_shift_range  = 0.10,
        height_shift_range = 0.10,
        zoom_range       = 0.10,
        horizontal_flip  = False,          # sign language is NOT symmetric
    )

    test_gen = ImageDataGenerator(
        rescale = 1.0 / 255,               # FIX 1 — same normalisation
    )

    train_batches = train_gen.flow_from_directory(
        directory  = train_path,
        target_size = (IMG_SIZE, IMG_SIZE),
        color_mode  = "rgb",               # loads grayscale JPGs as (H,W,3)
        class_mode  = "categorical",
        batch_size  = BATCH_SIZE,
        shuffle     = True,
    )

    test_batches = test_gen.flow_from_directory(
        directory  = test_path,
        target_size = (IMG_SIZE, IMG_SIZE),
        color_mode  = "rgb",
        class_mode  = "categorical",
        batch_size  = BATCH_SIZE,
        shuffle     = False,
    )

    return train_batches, test_batches


# ══════════════════════════════════════════════════════════════════════════════
# 4.  LABEL MAP
# ══════════════════════════════════════════════════════════════════════════════

def build_and_save_label_map(generator, save_path: str) -> dict:
    """
    ImageDataGenerator.class_indices  →  { folder_name: class_index }
    e.g. {"1": 0, "10": 1, "2": 2, "3": 3, …, "9": 9}   (alphabetical sort)

    We need the INVERSE:  { class_index: human_label }

    Folder names are digit strings "1"–"10".  Map them to words.
    The word mapping can be extended freely for alphabets, etc.
    """
    _folder_to_word = {
        "0": "Zero",   # in case someone adds zero later
        "1": "One",    "2": "Two",    "3": "Three",
        "4": "Four",   "5": "Five",   "6": "Six",
        "7": "Seven",  "8": "Eight",  "9": "Nine",
        "10": "Ten",
        # alphabet extensions (A-Z) can be added here
    }

    word_dict = {}
    for folder_name, class_idx in generator.class_indices.items():
        word = _folder_to_word.get(folder_name, folder_name)  # fallback = folder name
        word_dict[class_idx] = word

    payload = {
        "word_dict": {str(k): v for k, v in word_dict.items()},
        "class_indices": generator.class_indices,
        "img_size": IMG_SIZE,
        "num_classes": len(word_dict),
    }

    with open(save_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[INFO] label_map.json saved: {save_path}")
    print(f"[INFO] Mapping: {word_dict}")
    return word_dict


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

def build_model(num_classes: int = NUM_CLASSES) -> keras.Model:
    """
    Compact CNN well-suited to 64×64 binary/grayscale-as-RGB hand images.

    Architecture choices:
    • 3 Conv blocks with BatchNorm — stabilises training, reduces need for
      large learning-rate warmup.
    • MaxPool after each block — reduces spatial dims 64 → 32 → 16 → 8.
    • Two Dense layers with Dropout — prevents overfitting on small datasets.
    • Softmax output — consistent with categorical_crossentropy loss.

    Input shape: (None, 64, 64, 3)  — grayscale saved as RGB (R=G=B), rescaled [0,1].
    """
    model = Sequential([
        # ── Block 1 ──────────────────────────────────────────────────────────
        Conv2D(32, (3, 3), activation="relu", padding="same",
               input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),         # 64 → 32

        # ── Block 2 ──────────────────────────────────────────────────────────
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),         # 32 → 16

        # ── Block 3 ──────────────────────────────────────────────────────────
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),         # 16 → 8

        Flatten(),

        # ── Classifier head ──────────────────────────────────────────────────
        Dense(256, activation="relu"),
        Dropout(DROPOUT_RATE),
        Dense(128, activation="relu"),
        Dropout(DROPOUT_RATE),
        Dense(num_classes, activation="softmax"),
    ], name="gesture_cnn")

    model.compile(
        optimizer = Adam(learning_rate=LEARNING_RATE),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"],
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 6.  CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

def build_callbacks(model_path: str) -> list:
    return [
        ModelCheckpoint(
            filepath         = model_path,
            monitor          = "val_accuracy",
            save_best_only   = True,
            verbose          = 1,
        ),
        ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.3,
            patience = 3,
            min_lr   = 1e-5,
            verbose  = 1,
        ),
        EarlyStopping(
            monitor   = "val_loss",
            patience  = 5,
            verbose   = 1,
            restore_best_weights = True,
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 7.  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"],     label="train acc")
    axes[0].plot(history.history["val_accuracy"], label="val acc")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"],     label="train loss")
    axes[1].plot(history.history["val_loss"], label="val loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(_HERE, "training_history.png")
    plt.savefig(plot_path, dpi=100)
    print(f"[INFO] Training curves saved: {plot_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  Sign Language CNN Training  —  FINAL v4")
    print("=" * 70 + "\n")

    # ── Delete old model to prevent accidental use of stale weights ───────────
    for old in [MODEL_SAVE_PATH,
                MODEL_SAVE_PATH.replace(".h5", ".keras")]:
        if os.path.exists(old):
            os.remove(old)
            print(f"[INFO] Deleted old model: {old}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_path, test_path = get_dataset_paths()

    train_batches, test_batches = build_generators(train_path, test_path)

    print(f"\n[INFO] Found {train_batches.samples} training images "
          f"in {train_batches.num_classes} classes.")
    print(f"[INFO] Found {test_batches.samples}  test images.")
    print(f"[INFO] class_indices (alphabetical): {train_batches.class_indices}\n")

    # ── Save label map immediately (before training — safe even if training crashes)
    word_dict = build_and_save_label_map(train_batches, LABEL_MAP_PATH)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(num_classes=train_batches.num_classes)
    model.summary(line_length=80)

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n[INFO] Training started …\n")
    history = model.fit(
        train_batches,
        epochs            = EPOCHS,
        validation_data   = test_batches,
        callbacks         = build_callbacks(MODEL_SAVE_PATH),
        verbose           = 1,
    )

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\n[INFO] Loading best checkpoint for final evaluation …")
    best_model = keras.models.load_model(MODEL_SAVE_PATH)
    loss, acc  = best_model.evaluate(test_batches, verbose=0)
    print(f"[RESULT] Test loss: {loss:.4f}  |  Test accuracy: {acc*100:.2f}%\n")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_history(history)

    # ── Prediction sanity check ───────────────────────────────────────────────
    print("[INFO] Sample predictions on one test batch:")
    imgs, labels = next(iter(test_batches))
    preds        = best_model.predict(imgs, verbose=0)
    for i in range(min(10, len(preds))):
        pred_label  = word_dict.get(int(np.argmax(preds[i])),  str(np.argmax(preds[i])))
        true_label  = word_dict.get(int(np.argmax(labels[i])), str(np.argmax(labels[i])))
        match       = "✓" if pred_label == true_label else "✗"
        print(f"  {match}  predicted: {pred_label:<8}  actual: {true_label}")

    print(f"\n[DONE] Model saved:    {MODEL_SAVE_PATH}")
    print(f"[DONE] Label map saved: {LABEL_MAP_PATH}")
    print("\nNext step:  python model_for_gesture.py\n")


if __name__ == "__main__":
    main()