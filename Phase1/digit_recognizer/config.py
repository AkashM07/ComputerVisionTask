"""Shared paths and default settings for the digit recognizer.

Every other module imports from here, so a path or a default only has to
change in one place.
"""

from pathlib import Path

# Phase1/digit_recognizer/ -> repo root is two levels up.
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent

# image_path values in the annotation CSVs are relative to this folder.
DATASET_ROOT = REPO_ROOT / "DataCreation" / "a4_digits" / "a4_digits"

# results/ is already in .gitignore, so trained models and reports stay local.
RESULTS_DIR = REPO_ROOT / "results" / "digit_recognizer"
MODEL_PATH = RESULTS_DIR / "digit_svm.npz"

NUM_CLASSES = 10
SEED = 42

# --- Detection -------------------------------------------------------------
# The generator counts every pixel darker than pure white (< 255) as ink, so
# thresholding at 255 reproduces its ground-truth boxes exactly. A real photo
# needs a lower value or Otsu (pass threshold=None to detect_digit).
INK_THRESHOLD = 255
# Contours smaller than this fraction of the largest one are treated as noise.
MIN_CONTOUR_FRACTION = 0.0

# --- Crop normalisation ----------------------------------------------------
# Every detected digit is scaled (keeping its aspect ratio) to fit inside
# CANVAS_SIZE - 2 * CANVAS_PADDING pixels, then centred on a square canvas.
CANVAS_SIZE = 32
CANVAS_PADDING = 4

# --- HOG -------------------------------------------------------------------
HOG_BINS = 9
HOG_BLOCK_CELLS = 2

# --- Hyperparameter search (chosen on the validation split) ---------------
HOG_CELL_SIZES = (4, 8)
SVM_REGULARISATION = (1e-4, 1e-3, 1e-2)

# --- SVM training ----------------------------------------------------------
SVM_EPOCHS = 40
SVM_BATCH_SIZE = 128
SVM_LEARNING_RATE = 1e-2
SVM_LR_DECAY = 0.95
SVM_MARGIN = 1.0
