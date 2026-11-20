"""Evaluate the saved model once on the test split.

    python Phase1/digit_recognizer/evaluate.py [--save-images N]

Reports:
  * detector mean IoU and the share of boxes with IoU >= 0.5 / 0.9
  * end-to-end accuracy: detector box -> classifier (the real score)
  * recognition-only accuracy: ground-truth box -> classifier, to separate
    "found the wrong place" from "read the digit wrong"
  * a 10 x 10 confusion matrix (rows = true digit, columns = predicted)

Annotated pages are written to results/digit_recognizer/test_images/:
every mistake, plus the first N correct pages.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from config import MODEL_PATH, NUM_CLASSES, RESULTS_DIR
from data import load_split, read_gray
from detect import box_iou
from pipeline import DigitRecognizer, build_crops, draw_prediction


def confusion_matrix(true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    np.add.at(matrix, (true, predicted), 1)
    return matrix


def format_matrix(matrix: np.ndarray) -> str:
    header = "true\\pred " + " ".join(f"{d:>4}" for d in range(NUM_CLASSES))
    rows = [f"{d:>9} " + " ".join(f"{v:>4}" for v in matrix[d]) for d in range(NUM_CLASSES)]
    return "\n".join([header, *rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save-images", type=int, default=20, help="correct pages to save as examples")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(f"No model at {MODEL_PATH}. Run train.py first.")
    recogniser = DigitRecognizer.load(MODEL_PATH)

    samples = load_split("test")
    print("Reading test pages...")
    test = build_crops(samples)

    found = np.array([b is not None for b in test.predicted_boxes])
    ious = np.array(
        [0.0 if p is None else box_iou(p, t) for p, t in zip(test.predicted_boxes, test.true_boxes)]
    )
    end_to_end = recogniser.classify_crops(test.predicted_crops)
    end_to_end[~found] = -1  # no box -> counts as wrong
    recognition_only = recogniser.classify_crops(test.true_crops)

    e2e_acc = float(np.mean(end_to_end == test.labels))
    rec_acc = float(np.mean(recognition_only == test.labels))
    matrix = confusion_matrix(test.labels[found], end_to_end[found])

    report = "\n".join(
        [
            f"Test pages: {len(samples)}  (HOG cell size {recogniser.cell_size}, reg {recogniser.model.reg:g})",
            f"Pages with a detected digit: {found.sum()}",
            f"Detector mean IoU: {ious.mean():.4f}",
            f"Boxes with IoU >= 0.5: {np.mean(ious >= 0.5):.4f}   IoU >= 0.9: {np.mean(ious >= 0.9):.4f}",
            f"End-to-end accuracy (predicted box): {e2e_acc:.4f}",
            f"Recognition-only accuracy (ground-truth box): {rec_acc:.4f}",
            "",
            "Confusion matrix (end to end):",
            format_matrix(matrix),
        ]
    )
    print("\n" + report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "test_report.txt").write_text(report + "\n")

    image_dir = RESULTS_DIR / "test_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    correct_saved = 0
    for i, sample in enumerate(samples):
        wrong = end_to_end[i] != test.labels[i]
        if not wrong and correct_saved >= args.save_images:
            continue
        page = read_gray(sample.image_path)
        digit = None if end_to_end[i] < 0 else int(end_to_end[i])
        annotated = draw_prediction(page, test.predicted_boxes[i], digit, true_box=sample.box)
        prefix = f"WRONG_true{sample.digit}_pred{digit}_" if wrong else "ok_"
        cv2.imwrite(str(image_dir / f"{prefix}{sample.image_path.name}"), annotated)
        correct_saved += not wrong
    print(f"\nReport and annotated pages saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
