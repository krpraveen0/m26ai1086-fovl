"""A small, reproducible binary flower classifier.

The notebook introduces the mathematics of a single sigmoid neuron.  This
module packages the same model so it can be trained and evaluated repeatedly
without relying on notebook state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_DATA = np.array(
    [
        [3.0, 1.5, 1.0],
        [2.0, 1.0, 0.0],
        [4.0, 1.5, 1.0],
        [3.0, 1.0, 0.0],
        [3.5, 0.5, 1.0],
        [2.0, 0.5, 0.0],
        [5.5, 1.0, 1.0],
        [1.0, 1.0, 0.0],
    ],
    dtype=float,
)


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities while avoiding exponential overflow."""
    values = np.asarray(values, dtype=float)
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


@dataclass
class TrainingHistory:
    """Loss values recorded during training."""

    losses: list[float]


class FlowerClassifier:
    """A single sigmoid neuron trained with batch gradient descent."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        epochs: int = 2_000,
        seed: int = 7,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        if epochs <= 0:
            raise ValueError("epochs must be greater than zero")
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.weights: np.ndarray | None = None
        self.bias: float | None = None
        self.history: TrainingHistory | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> TrainingHistory:
        """Fit the classifier and return mean squared error by epoch."""
        x, y = self._validate_training_data(features, targets)
        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(0.0, 0.5, size=x.shape[1])
        self.bias = float(rng.normal(0.0, 0.5))
        losses: list[float] = []

        for _ in range(self.epochs):
            probabilities = self.predict_proba(x)
            error = probabilities - y
            losses.append(float(np.mean(error**2)))

            gradient = 2.0 * error * probabilities * (1.0 - probabilities)
            self.weights -= self.learning_rate * (x.T @ gradient / len(x))
            self.bias -= self.learning_rate * float(np.mean(gradient))

        self.history = TrainingHistory(losses=losses)
        return self.history

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return the probability that each row belongs to class 1."""
        if self.weights is None or self.bias is None:
            raise RuntimeError("fit must be called before making predictions")
        x = self._validate_features(features)
        return sigmoid(x @ self.weights + self.bias)

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions using the supplied probability threshold."""
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between zero and one")
        return (self.predict_proba(features) >= threshold).astype(int)

    def evaluate(self, features: np.ndarray, targets: np.ndarray) -> dict[str, float]:
        """Return accuracy, precision, recall, and mean squared error."""
        x = self._validate_features(features)
        y = self._validate_targets(targets, len(x))
        probabilities = self.predict_proba(x)
        predictions = (probabilities >= 0.5).astype(int)
        true_positive = np.sum((predictions == 1) & (y == 1))
        predicted_positive = np.sum(predictions == 1)
        actual_positive = np.sum(y == 1)
        return {
            "accuracy": float(np.mean(predictions == y)),
            "precision": float(true_positive / predicted_positive)
            if predicted_positive
            else 0.0,
            "recall": float(true_positive / actual_positive) if actual_positive else 0.0,
            "mse": float(np.mean((probabilities - y) ** 2)),
        }

    def decision_boundary(self, x_values: np.ndarray) -> np.ndarray:
        """Return width values on the 0.5 probability boundary."""
        if self.weights is None or self.bias is None:
            raise RuntimeError("fit must be called before calculating the boundary")
        if np.isclose(self.weights[1], 0.0):
            raise RuntimeError("the learned boundary is vertical")
        x_values = np.asarray(x_values, dtype=float)
        return -(self.weights[0] * x_values + self.bias) / self.weights[1]

    @staticmethod
    def _validate_features(features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError("features must be a 2D array with two columns")
        if not np.all(np.isfinite(x)):
            raise ValueError("features must contain only finite values")
        return x

    @classmethod
    def _validate_training_data(
        cls, features: np.ndarray, targets: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        x = cls._validate_features(features)
        y = cls._validate_targets(targets, len(x))
        return x, y

    @staticmethod
    def _validate_targets(targets: np.ndarray, expected_length: int) -> np.ndarray:
        y = np.asarray(targets, dtype=float).reshape(-1)
        if len(y) != expected_length or not np.all(np.isin(y, [0.0, 1.0])):
            raise ValueError("targets must contain one binary label per feature row")
        return y

