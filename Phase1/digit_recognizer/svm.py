"""Stage 3: a multiclass linear SVM written from scratch in NumPy.

The model keeps one weight vector per digit. For a feature vector x the score
of digit k is  w_k . x + b_k,  and the prediction is the digit with the
highest score.

Training minimises the multiclass hinge loss (Weston-Watkins):

    loss(x, y) = sum over k != y of  max(0, score_k - score_y + margin)
               + reg * ||W||^2

In words: the correct digit's score must beat every other digit's score by at
least `margin`; each digit that comes closer than that adds to the loss.
We minimise it with mini-batch stochastic gradient descent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import (
    NUM_CLASSES,
    SEED,
    SVM_BATCH_SIZE,
    SVM_EPOCHS,
    SVM_LEARNING_RATE,
    SVM_LR_DECAY,
    SVM_MARGIN,
)


class LinearSVM:
    def __init__(
        self,
        reg: float,
        num_classes: int = NUM_CLASSES,
        epochs: int = SVM_EPOCHS,
        batch_size: int = SVM_BATCH_SIZE,
        learning_rate: float = SVM_LEARNING_RATE,
        lr_decay: float = SVM_LR_DECAY,
        margin: float = SVM_MARGIN,
        seed: int = SEED,
    ) -> None:
        self.reg = reg
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.margin = margin
        self.seed = seed
        # Set by fit() / load().
        self.weights: np.ndarray | None = None  # (feature_dim + 1, num_classes)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.loss_history: list[float] = []

    # --- preprocessing ------------------------------------------------------
    def _prepare(self, features: np.ndarray) -> np.ndarray:
        """Standardise each feature, then append a constant 1 for the bias."""
        standardised = (features - self.mean) / self.std
        ones = np.ones((standardised.shape[0], 1), dtype=standardised.dtype)
        return np.hstack([standardised, ones])

    # --- training -----------------------------------------------------------
    def _loss_and_gradient(self, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        n = x.shape[0]
        scores = x @ self.weights  # (n, classes)
        correct = scores[np.arange(n), y][:, None]
        margins = np.maximum(0.0, scores - correct + self.margin)
        margins[np.arange(n), y] = 0.0

        # The bias row (last) is not regularised.
        w_no_bias = self.weights[:-1]
        loss = margins.sum() / n + self.reg * np.sum(w_no_bias**2)

        # Each violating class pushes its own weights toward -x and pulls the
        # correct class's weights toward +x.
        indicator = (margins > 0).astype(x.dtype)
        indicator[np.arange(n), y] = -indicator.sum(axis=1)
        gradient = x.T @ indicator / n
        gradient[:-1] += 2.0 * self.reg * w_no_bias
        return float(loss), gradient

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "LinearSVM":
        rng = np.random.default_rng(self.seed)
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0) + 1e-6
        x = self._prepare(features)
        labels = labels.astype(int)

        self.weights = 0.001 * rng.standard_normal((x.shape[1], self.num_classes))
        self.loss_history = []
        learning_rate = self.learning_rate

        for _ in range(self.epochs):
            order = rng.permutation(len(x))
            epoch_loss = 0.0
            for start in range(0, len(x), self.batch_size):
                batch = order[start : start + self.batch_size]
                loss, gradient = self._loss_and_gradient(x[batch], labels[batch])
                self.weights -= learning_rate * gradient
                epoch_loss += loss * len(batch)
            self.loss_history.append(epoch_loss / len(x))
            learning_rate *= self.lr_decay
        return self

    # --- inference ----------------------------------------------------------
    def scores(self, features: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("model is not trained; call fit() or load() first")
        return self._prepare(np.atleast_2d(features)) @ self.weights

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.scores(features).argmax(axis=1)

    # --- persistence --------------------------------------------------------
    def save(self, path: Path, extra: dict | None = None) -> None:
        """Save weights plus any settings needed to rebuild the features."""
        path.parent.mkdir(parents=True, exist_ok=True)
        settings = {"reg": self.reg, "num_classes": self.num_classes, **(extra or {})}
        np.savez(
            path,
            weights=self.weights,
            mean=self.mean,
            std=self.std,
            settings=json.dumps(settings),
        )

    @classmethod
    def load(cls, path: Path) -> tuple["LinearSVM", dict]:
        data = np.load(path)
        settings = json.loads(str(data["settings"]))
        model = cls(reg=settings["reg"], num_classes=settings["num_classes"])
        model.weights = data["weights"]
        model.mean = data["mean"]
        model.std = data["std"]
        return model, settings
