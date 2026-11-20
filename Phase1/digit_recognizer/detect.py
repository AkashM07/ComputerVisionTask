"""Stage 1: find where the digit is on the page.

Same idea as Phase1/object_detection_p1_am.py (threshold -> contours ->
bounding box), with two changes:

* The box is the union of all contours, not only the largest one. Digits
  in some fonts break into pieces (the dot-like ends of a serif "1", or a
  thin stroke that antialiasing splits), and taking only the largest piece
  would cut them off.
* The threshold is configurable, with Otsu as the fallback for real photos
  where "paper" is not exactly 255.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import INK_THRESHOLD, MIN_CONTOUR_FRACTION

Box = tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max), max exclusive


def ink_mask(gray: np.ndarray, threshold: int | None = INK_THRESHOLD) -> np.ndarray:
    """Return a uint8 mask where ink is 255 and paper is 0.

    threshold=N marks every pixel darker than N as ink.
    threshold=None picks the cut automatically with Otsu's method.
    """
    if threshold is None:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return mask
    return np.where(gray < threshold, 255, 0).astype(np.uint8)


def detect_digit(
    gray: np.ndarray,
    threshold: int | None = INK_THRESHOLD,
    min_contour_fraction: float = MIN_CONTOUR_FRACTION,
) -> Box | None:
    """Return the digit's bounding box, or None when the page has no ink."""
    mask = ink_mask(gray, threshold)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    rects = [cv2.boundingRect(c) for c in contours]
    largest_area = max(w * h for _, _, w, h in rects)
    keep = [r for r in rects if r[2] * r[3] >= min_contour_fraction * largest_area]

    x_min = min(x for x, _, _, _ in keep)
    y_min = min(y for _, y, _, _ in keep)
    x_max = max(x + w for x, _, w, _ in keep)
    y_max = max(y + h for _, y, _, h in keep)
    return (x_min, y_min, x_max, y_max)


def box_iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes with exclusive max corners."""
    inter_w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = inter_w * inter_h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union
