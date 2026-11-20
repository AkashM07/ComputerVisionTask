"""Run the trained model on your own images.

    python Phase1/digit_recognizer/predict.py Phase1/test_data_p1
    python Phase1/digit_recognizer/predict.py some_photo.png --out my_results

Accepts image files and/or folders. Prints the digit for each image and saves
a copy with the box and label drawn on it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from config import MODEL_PATH, RESULTS_DIR
from data import read_gray
from pipeline import DigitRecognizer, draw_prediction

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def collect_images(inputs: list[str]) -> list[Path]:
    paths = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES))
        elif path.is_file():
            paths.append(path)
        else:
            print(f"Skipping {path}: not found")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="image files or folders")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "predictions", help="where to save annotated images")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"No model at {args.model}. Run train.py first.")
    recogniser = DigitRecognizer.load(args.model)
    args.out.mkdir(parents=True, exist_ok=True)

    for path in collect_images(args.inputs):
        page = read_gray(path)
        box, digit = recogniser.recognise(page)
        print(f"{path.name}: {'no digit found' if digit is None else digit}  box={box}")
        cv2.imwrite(str(args.out / path.name), draw_prediction(page, box, digit))
    print(f"Annotated images saved to {args.out}")


if __name__ == "__main__":
    main()
