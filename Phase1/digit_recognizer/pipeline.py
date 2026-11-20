"""Glue between the stages: page -> box -> normalised crop -> HOG -> digit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import CANVAS_SIZE
from data import Sample, read_gray
from detect import Box, detect_digit
from features import hog_features, hog_matrix, normalize_crop
from svm import LinearSVM


@dataclass
class SplitCrops:
    """Normalised crops for one split, computed once and reused for every
    HOG setting tried during the hyperparameter search."""

    labels: np.ndarray  # (n,)
    true_boxes: list[Box]
    predicted_boxes: list[Box | None]
    true_crops: np.ndarray  # crops cut with the ground-truth box
    predicted_crops: np.ndarray  # crops cut with the detector's box (blank if none)


def build_crops(samples: list[Sample], detect: bool = True) -> SplitCrops:
    """Read each page once and cut its digit out with both kinds of box.

    detect=False skips the detector (used for train, where we only learn from
    ground-truth crops).
    """
    labels, true_boxes, predicted_boxes, true_crops, predicted_crops = [], [], [], [], []
    blank = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)

    for index, sample in enumerate(samples, start=1):
        page = read_gray(sample.image_path)
        labels.append(sample.digit)
        true_boxes.append(sample.box)
        true_crops.append(normalize_crop(page, sample.box))

        if detect:
            box = detect_digit(page)
            predicted_boxes.append(box)
            predicted_crops.append(blank if box is None else normalize_crop(page, box))

        if index % 500 == 0:
            print(f"  read {index}/{len(samples)} pages")

    return SplitCrops(
        labels=np.array(labels),
        true_boxes=true_boxes,
        predicted_boxes=predicted_boxes,
        true_crops=np.stack(true_crops),
        predicted_crops=np.stack(predicted_crops) if predicted_crops else np.empty((0,)),
    )


class DigitRecognizer:
    """A trained detector + classifier, ready to run on new pages."""

    def __init__(self, model: LinearSVM, cell_size: int) -> None:
        self.model = model
        self.cell_size = cell_size

    @classmethod
    def load(cls, path: Path) -> "DigitRecognizer":
        model, settings = LinearSVM.load(path)
        return cls(model, settings["cell_size"])

    def classify_crops(self, crops: np.ndarray) -> np.ndarray:
        return self.model.predict(hog_matrix(crops, self.cell_size))

    def recognise(self, page: np.ndarray) -> tuple[Box | None, int | None]:
        """Find and read the digit on one grayscale page."""
        box = detect_digit(page)
        if box is None:
            return None, None
        crop = normalize_crop(page, box)
        digit = int(self.model.predict(hog_features(crop, self.cell_size))[0])
        return box, digit


def draw_prediction(
    page: np.ndarray,
    box: Box | None,
    digit: int | None,
    true_box: Box | None = None,
) -> np.ndarray:
    """Return a colour copy of the page with the boxes and the label drawn.

    Green = predicted box, blue = ground truth (when given).
    """
    canvas = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
    if true_box is not None:
        # Boxes are max-exclusive; OpenCV rectangles are inclusive.
        cv2.rectangle(canvas, true_box[:2], (true_box[2] - 1, true_box[3] - 1), (255, 0, 0), 1)
    if box is None:
        cv2.putText(canvas, "no digit found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return canvas

    cv2.rectangle(canvas, box[:2], (box[2] - 1, box[3] - 1), (0, 200, 0), 2)
    label_y = box[1] - 8 if box[1] > 30 else box[3] + 28
    cv2.putText(canvas, str(digit), (box[0], label_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)
    return canvas
