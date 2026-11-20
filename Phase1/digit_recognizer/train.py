"""Train the digit classifier and pick its settings on the validation split.

    python Phase1/digit_recognizer/train.py

1. Cut every training digit out with its ground-truth box.
2. For each (HOG cell size, SVM regularisation) pair: train on train, then
   score on validation end to end (the detector's own box, never the
   ground-truth one).
3. Retrain the best pair on train and save it to results/digit_recognizer/.

The test split is not touched here; evaluate.py runs it once at the end.
"""

from __future__ import annotations

import json

import numpy as np

from config import HOG_CELL_SIZES, MODEL_PATH, RESULTS_DIR, SVM_REGULARISATION
from data import load_split
from detect import box_iou
from features import hog_matrix
from pipeline import build_crops
from svm import LinearSVM


def main() -> None:
    print("Reading train pages...")
    train = build_crops(load_split("train"), detect=False)
    print("Reading validation pages...")
    validation = build_crops(load_split("validation"))

    ious = [
        0.0 if p is None else box_iou(p, t)
        for p, t in zip(validation.predicted_boxes, validation.true_boxes)
    ]
    print(f"Validation detector mean IoU: {np.mean(ious):.4f}")

    results = []
    best = None
    for cell_size in HOG_CELL_SIZES:
        train_x = hog_matrix(train.true_crops, cell_size)
        val_x = hog_matrix(validation.predicted_crops, cell_size)
        for reg in SVM_REGULARISATION:
            model = LinearSVM(reg=reg).fit(train_x, train.labels)
            train_acc = float(np.mean(model.predict(train_x) == train.labels))
            val_acc = float(np.mean(model.predict(val_x) == validation.labels))
            print(
                f"cell={cell_size} reg={reg:g}  features={train_x.shape[1]}  "
                f"train acc={train_acc:.4f}  validation acc={val_acc:.4f}  "
                f"final loss={model.loss_history[-1]:.4f}"
            )
            row = {"cell_size": cell_size, "reg": reg, "train_acc": train_acc, "val_acc": val_acc}
            results.append(row)
            # Strict ">" so a tie keeps the setting tried first.
            if best is None or val_acc > best["val_acc"]:
                best = row

    print(f"\nBest: cell={best['cell_size']} reg={best['reg']:g} validation acc={best['val_acc']:.4f}")

    final_x = hog_matrix(train.true_crops, best["cell_size"])
    model = LinearSVM(reg=best["reg"]).fit(final_x, train.labels)
    model.save(MODEL_PATH, extra={"cell_size": best["cell_size"]})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "search.json").write_text(json.dumps({"best": best, "all": results}, indent=2))
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
