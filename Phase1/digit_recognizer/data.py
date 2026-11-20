"""Read the A4 digit dataset: annotation rows and page images."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import DATASET_ROOT

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class Sample:
    """One page and its ground truth.

    The box uses the dataset's convention: top-left origin, x_max / y_max
    exclusive, so the crop is page[y_min:y_max, x_min:x_max].
    """

    image_path: Path
    digit: int
    box: tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)


def load_split(split: str, dataset_root: Path = DATASET_ROOT) -> list[Sample]:
    """Return every sample in one split, in CSV order."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    csv_path = dataset_root / split / "annotations.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Unzip DataCreation/a4_digits.zip or run "
            "DataCreation/generate_digit_dataset.py first."
        )

    samples = []
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            box = (
                int(row["x_min"]),
                int(row["y_min"]),
                int(row["x_max"]),
                int(row["y_max"]),
            )
            samples.append(
                Sample(
                    image_path=dataset_root / row["image_path"],
                    digit=int(row["digit"]),
                    box=box,
                )
            )
    return samples


def read_gray(path: Path) -> np.ndarray:
    """Read an image as a single-channel uint8 array (white paper = 255)."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image {path}")
    return image
