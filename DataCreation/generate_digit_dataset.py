"""Generate 5,000 synthetic A4-page digit images using OpenCV 4.x.

Colab: import this file, then call generate_dataset('/content/a4_digits').
Terminal: python generate_digit_dataset.py --output a4_digits --seed 42
This creates data and annotations only; it does not train a classifier.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import zipfile

import cv2
import numpy as np

PAGE_WIDTH, PAGE_HEIGHT = 595, 842
MARGIN = 16
MIN_DIGIT_HEIGHT, MAX_DIGIT_HEIGHT = 40, 300
SPLIT_COUNTS = {"train": 3500, "validation": 500, "test": 1000}
FONTS = {
    "simplex": cv2.FONT_HERSHEY_SIMPLEX,
    "duplex": cv2.FONT_HERSHEY_DUPLEX,
    "complex": cv2.FONT_HERSHEY_COMPLEX,
    "triplex": cv2.FONT_HERSHEY_TRIPLEX,
    "plain": cv2.FONT_HERSHEY_PLAIN,
}
FIELDS = ["image_path", "split", "digit", "x_min", "y_min", "x_max", "y_max",
          "bbox_width", "bbox_height", "page_width", "page_height", "font",
          "font_scale", "stroke_thickness", "target_height", "pixel_sha256"]


def render_digit(digit, font, target_height, thickness):
    """Draw with padding, then crop to ALL nonwhite pixels, including AA edges."""
    (_, base_height), _ = cv2.getTextSize(str(digit), font, 1.0, thickness)
    scale = target_height / max(base_height, 1)
    (width, height), baseline = cv2.getTextSize(str(digit), font, scale, thickness)
    pad = max(32, math.ceil(8 * scale) + 3 * thickness)
    canvas = np.full((height + baseline + 2 * pad, width + 2 * pad), 255, np.uint8)
    cv2.putText(canvas, str(digit), (pad, pad + height), font, scale, 0,
                thickness, cv2.LINE_AA)
    yy, xx = np.nonzero(canvas < 255)
    if not len(xx):
        raise RuntimeError("Rendered an empty digit")
    # Detect accidental clipping on the temporary rendering canvas.
    if xx.min() == 0 or yy.min() == 0 or xx.max() == canvas.shape[1]-1 or yy.max() == canvas.shape[0]-1:
        raise RuntimeError("Digit touches rendering canvas boundary")
    crop = canvas[yy.min():yy.max()+1, xx.min():xx.max()+1].copy()
    return crop, scale


def make_preview(root, examples):
    """Overview of one example per digit, from each split; no boxes on data."""
    thumb_w, thumb_h, header = 119, 168, 26
    preview = np.full((3 * (thumb_h + header), 10 * thumb_w), 235, np.uint8)
    for row_i, split in enumerate(SPLIT_COUNTS):
        for digit in range(10):
            image = cv2.imread(str(root / examples[(split, digit)]), cv2.IMREAD_GRAYSCALE)
            thumb = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            y, x = row_i * (thumb_h + header), digit * thumb_w
            preview[y+header:y+header+thumb_h, x:x+thumb_w] = thumb
            cv2.putText(preview, f"{split}: {digit}", (x+3, y+17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, 0, 1, cv2.LINE_AA)
    if not cv2.imwrite(str(root / "preview.png"), preview):
        raise IOError("Could not write preview")


def generate_dataset(output_dir="a4_digits", seed=42):
    """Generate balanced, disjoint splits and return the ZIP's absolute Path.

    Refuses to overwrite an existing directory or ZIP. To regenerate, choose
    a new output name. Same seed + settings + library versions reproduce data.
    """
    root = Path(output_dir).expanduser().resolve()
    archive = root.parent / (root.name + ".zip")
    if root.exists() or archive.exists():
        raise FileExistsError(f"Choose a new output name; {root} or {archive} exists")
    if not (0 < MIN_DIGIT_HEIGHT <= MAX_DIGIT_HEIGHT):
        raise ValueError("Invalid digit height limits")
    if MARGIN < 1:
        raise ValueError("Margin must be at least one pixel")
    for split, count in SPLIT_COUNTS.items():
        if count <= 0 or count % 10:
            raise ValueError(f"{split} count must be a positive multiple of 10")
    root.mkdir(parents=True)
    streams = np.random.SeedSequence(seed).spawn(len(SPLIT_COUNTS))
    seen = set()  # Across every split: no identical page pixels.
    rows, examples = [], {}
    font_names = list(FONTS)

    for (split, count), stream in zip(SPLIT_COUNTS.items(), streams):
        rng = np.random.default_rng(stream)
        (root / split / "images").mkdir(parents=True)
        # Shuffle IDs so filenames do not encode the digit label.
        labels = np.repeat(np.arange(10), count // 10)
        rng.shuffle(labels)
        split_rows = []
        for index, digit in enumerate(labels):
            for attempt in range(100):
                font_name = font_names[int(rng.integers(len(font_names)))]
                target = int(rng.integers(MIN_DIGIT_HEIGHT, MAX_DIGIT_HEIGHT + 1))
                thickness = max(1, int(round(target * rng.uniform(0.008, 0.025))))
                crop, scale = render_digit(int(digit), FONTS[font_name], target, thickness)
                h, w = crop.shape
                if w > PAGE_WIDTH-2*MARGIN or h > PAGE_HEIGHT-2*MARGIN:
                    continue
                x = int(rng.integers(MARGIN, PAGE_WIDTH-MARGIN-w+1))
                y = int(rng.integers(MARGIN, PAGE_HEIGHT-MARGIN-h+1))
                page = np.full((PAGE_HEIGHT, PAGE_WIDTH), 255, np.uint8)
                page[y:y+h, x:x+w] = crop
                digest = hashlib.sha256(page.tobytes()).hexdigest()
                if digest not in seen:
                    seen.add(digest)
                    break
            else:
                raise RuntimeError("Could not produce a valid, unique image")
            relative = f"{split}/images/{index:05d}.png"
            if not cv2.imwrite(str(root / relative), page, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
                raise IOError(f"Could not write {relative}")
            # Maxima are EXCLUSIVE, matching NumPy slicing page[y_min:y_max, x_min:x_max].
            row = dict(zip(FIELDS, [relative, split, int(digit), x, y, x+w, y+h,
                                   w, h, PAGE_WIDTH, PAGE_HEIGHT, font_name, scale,
                                   thickness, target, digest]))
            split_rows.append(row)
            examples.setdefault((split, int(digit)), relative)
        rows.extend(split_rows)
        with (root / split / "annotations.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(split_rows)
        print(f"{split}: {count:,} images ({count//10} per digit)")

    metadata = {
        "seed": seed, "splits": SPLIT_COUNTS, "image_count": len(rows),
        "page_width": PAGE_WIDTH, "page_height": PAGE_HEIGHT,
        "a4_note": "Portrait A4 proportions, approximately 210:297; no physical DPI specified.",
        "margin_pixels": MARGIN, "target_digit_height_range": [MIN_DIGIT_HEIGHT, MAX_DIGIT_HEIGHT],
        "height_note": "Requested OpenCV text height; actual tight ink height is bbox_height.",
        "fonts": font_names, "opencv_version": cv2.__version__, "numpy_version": np.__version__,
        "bbox_convention": "Top-left origin; x right, y down; x_max/y_max exclusive. All pixels <255 count as ink.",
        "unique_page_hashes": len(seen),
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (root / "README.md").write_text(DATASET_README, encoding="utf-8")
    make_preview(root, examples)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                z.write(path, arcname=f"{root.name}/{path.relative_to(root).as_posix()}")
    print(f"Dataset ZIP: {archive} ({archive.stat().st_size/1024**2:.1f} MiB)")
    return archive


DATASET_README = """# Synthetic A4 digit dataset

5,000 single-channel grayscale PNG pages, 595 pixels wide by 842 high.
Each contains exactly one upright printed digit (0-9), black on white.
The page fills the image; there is no desk, paper border, rotation, noise,
shadow, handwriting, or perspective distortion. Edges are antialiased.

| Split | Images | Images per digit |
|---|---:|---:|
| train | 3,500 | 350 |
| validation | 500 | 50 |
| test | 1,000 | 100 |

Each split contains images/ and annotations.csv. image_path in the CSV is
relative to the dataset root, not the CSV's parent. Filenames do not encode
class labels. preview.png is an overview and is NOT a training example.
metadata.json records settings and generation versions.

## Annotation coordinates
Origin is the upper-left image corner. x increases right; y increases down.
Boxes tightly enclose all nonwhite pixels, including antialiased edge pixels.
x_min/y_min are inclusive; x_max/y_max are exclusive. Thus:
width = x_max - x_min, height = y_max - y_min.
Crop with page[y_min:y_max, x_min:x_max]. To draw an inclusive OpenCV
rectangle, use (x_min, y_min), (x_max-1, y_max-1).
The page has at least 16 white pixels of margin around every digit.
font_scale, font, stroke_thickness and target_height describe generation;
bbox_height records the actual ink height. pixel_sha256 hashes decoded
page pixels and is unique across all splits.

## Intended next steps
Locate digits using thresholding and contours; preserve aspect ratio when
resizing crops onto a padded fixed-size canvas; extract HOG features; train
an SVM using train only. Select preprocessing and model settings using
validation. Evaluate the final fixed pipeline once on test. Report digit
accuracy and confusion matrix, plus bounding-box intersection over union.
Never give the recognition model the true digit or ground-truth box at
end-to-end test time: it must use its predicted box. Ground-truth crops can
be used separately to isolate recognition performance.

Splits share the same font/size distributions and contain no identical pages.
That measures performance on this synthetic generator, not on handwriting,
photographs, unseen fonts, or real-world documents. Later add separate
challenge datasets for those conditions. More synthetic images alone do not
establish real-world performance.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="a4_digits", help="New output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args.output, args.seed)
