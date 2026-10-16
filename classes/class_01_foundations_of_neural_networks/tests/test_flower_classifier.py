import numpy as np
import pytest

from classes.class_01_foundations_of_neural_networks.src.flower_classifier import (
    DEFAULT_DATA,
    FlowerClassifier,
    sigmoid,
)


def test_sigmoid_is_stable_for_large_logits():
    probabilities = sigmoid(np.array([-1_000.0, 0.0, 1_000.0]))

    assert np.allclose(probabilities, [0.0, 0.5, 1.0])


def test_classifier_learns_the_training_examples():
    model = FlowerClassifier(learning_rate=0.1, epochs=2_000, seed=7)
    features, targets = DEFAULT_DATA[:, :2], DEFAULT_DATA[:, 2]

    history = model.fit(features, targets)
    metrics = model.evaluate(features, targets)

    assert len(history.losses) == 2_000
    assert history.losses[-1] < history.losses[0]
    assert metrics["accuracy"] >= 0.875
    assert model.predict([[4.5, 1.0]])[0] == 1


def test_training_is_reproducible_for_a_seed():
    features, targets = DEFAULT_DATA[:, :2], DEFAULT_DATA[:, 2]
    first = FlowerClassifier(seed=11)
    second = FlowerClassifier(seed=11)

    first.fit(features, targets)
    second.fit(features, targets)

    np.testing.assert_allclose(first.weights, second.weights)
    assert first.bias == second.bias


@pytest.mark.parametrize(
    ("features", "targets", "message"),
    [
        ([[1.0]], [0], "two columns"),
        ([[1.0, np.nan]], [0], "finite"),
        ([[1.0, 1.0]], [2], "binary"),
    ],
)
def test_invalid_training_data_is_rejected(features, targets, message):
    with pytest.raises(ValueError, match=message):
        FlowerClassifier().fit(features, targets)
