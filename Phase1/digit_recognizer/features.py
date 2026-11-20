"""Stage 2: turn a detected digit into a fixed-length feature vector.

Two steps:

1. normalize_crop  - cut the box out of the page, scale it to a small square
                     canvas without stretching it, and centre it. After this a
                     tiny "7" in a corner and a huge "7" in the middle look
                     the same, which is what lets one classifier handle both.
2. hog_features    - Histogram of Oriented Gradients, written by hand with
                     NumPy (no cv2.HOGDescriptor / skimage) so every step is
                     visible. HOG describes the shape by the directions its
                     edges point in, measured in small cells.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import CANVAS_PADDING, CANVAS_SIZE, HOG_BINS, HOG_BLOCK_CELLS

Box = tuple[int, int, int, int]


def normalize_crop(
    gray: np.ndarray,
    box: Box,
    canvas_size: int = CANVAS_SIZE,
    padding: int = CANVAS_PADDING,
) -> np.ndarray:
    """Return a canvas_size x canvas_size float32 image, ink = 1, paper = 0."""
    x_min, y_min, x_max, y_max = box
    crop = gray[y_min:y_max, x_min:x_max]
    # Invert so ink is bright: gradients and zero-padding both behave better
    # when the background is 0.
    ink = (255.0 - crop.astype(np.float32)) / 255.0

    height, width = ink.shape
    inner = canvas_size - 2 * padding
    scale = inner / max(height, width)
    new_w = max(1, round(width * scale))
    new_h = max(1, round(height * scale))
    # INTER_AREA averages pixels when shrinking (no aliasing);
    # INTER_LINEAR is the sensible choice when enlarging a small digit.
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(ink, (new_w, new_h), interpolation=interpolation)

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    top = (canvas_size - new_h) // 2
    left = (canvas_size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def hog_features(
    image: np.ndarray,
    cell_size: int,
    bins: int = HOG_BINS,
    block_cells: int = HOG_BLOCK_CELLS,
) -> np.ndarray:
    """Compute a HOG descriptor for one square float image.

    Steps:
      1. Gradients with the [-1, 0, 1] filter in x and y.
      2. Magnitude and *unsigned* orientation (0-180 degrees): a stroke's
         edge looks the same whether we cross it dark->light or light->dark.
      3. Each pixel votes its magnitude into the orientation histogram of its
         cell, split between the two nearest bins (so 19 degrees does not
         land entirely in the 20-degree bin).
      4. Cells are grouped into overlapping blocks and each block is
         L2-normalised, clipped at 0.2 and renormalised ("L2-Hys"). This
         makes the descriptor insensitive to stroke thickness / contrast.
    """
    size = image.shape[0]
    if image.shape != (size, size):
        raise ValueError(f"expected a square image, got {image.shape}")
    if size % cell_size:
        raise ValueError(f"image size {size} is not divisible by cell size {cell_size}")

    # 1. Gradients. Edge padding avoids fake gradients at the border.
    padded = np.pad(image, 1, mode="edge")
    gx = padded[1:-1, 2:] - padded[1:-1, :-2]
    gy = padded[2:, 1:-1] - padded[:-2, 1:-1]

    # 2. Magnitude and unsigned orientation in [0, 180).
    magnitude = np.hypot(gx, gy)
    orientation = np.rad2deg(np.arctan2(gy, gx)) % 180.0

    # 3. Soft-binned cell histograms.
    bin_width = 180.0 / bins
    position = orientation / bin_width - 0.5  # bin centres sit at 0.5, 1.5, ...
    lower = np.floor(position)
    upper_weight = position - lower
    lower_bin = lower.astype(int) % bins
    upper_bin = (lower_bin + 1) % bins

    cells = size // cell_size
    cell_index = np.arange(size) // cell_size
    cell_y, cell_x = np.meshgrid(cell_index, cell_index, indexing="ij")

    histogram = np.zeros((cells, cells, bins), dtype=np.float64)
    np.add.at(histogram, (cell_y, cell_x, lower_bin), magnitude * (1.0 - upper_weight))
    np.add.at(histogram, (cell_y, cell_x, upper_bin), magnitude * upper_weight)

    # 4. Overlapping blocks of block_cells x block_cells cells, stride 1 cell.
    blocks_per_side = cells - block_cells + 1
    if blocks_per_side < 1:
        raise ValueError("cell size too large: fewer cells than one block")
    blocks = []
    for by in range(blocks_per_side):
        for bx in range(blocks_per_side):
            block = histogram[by : by + block_cells, bx : bx + block_cells].ravel()
            block = block / np.sqrt(np.sum(block**2) + 1e-6)
            block = np.minimum(block, 0.2)
            block = block / np.sqrt(np.sum(block**2) + 1e-6)
            blocks.append(block)
    return np.concatenate(blocks).astype(np.float32)


def hog_matrix(crops: np.ndarray, cell_size: int) -> np.ndarray:
    """HOG for a stack of crops, shape (n, size, size) -> (n, feature_dim)."""
    return np.stack([hog_features(crop, cell_size) for crop in crops])
