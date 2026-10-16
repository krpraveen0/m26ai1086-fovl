# Class 01: Foundations of neural networks

**Author:** Praveen Kumar

## Learning goals

- Understand a single neuron as a weighted sum followed by an activation.
- Implement the sigmoid function and its derivative.
- Follow gradient descent and backpropagation for a binary prediction.
- Inspect loss and a learned decision boundary.

## Materials

- [Intro to neural networks notes](./notes/intro-to-neural-networks.pdf)
- [Flower neuron notebook](./notebooks/flowers.ipynb)
- [Reusable flower classifier](./src/flower_classifier.py)
- [Classifier tests](./tests/test_flower_classifier.py)

## Project status

The notebook is the original classroom walkthrough. The Python module
packages the same idea into a deterministic, testable experiment. The toy
dataset is intentionally small and is not perfectly linearly separable, so
the model is useful for understanding the method rather than measuring
real-world flower recognition performance.

## Follow-up ideas

1. Add a train/test split and compare metrics across several seeds.
2. Replace the toy data with the Iris dataset.
3. Compare mean-squared-error training with binary cross-entropy.
4. Add feature normalization and inspect how it changes convergence.

## Running this class

From the repository root:

```bash
uv run pytest classes/class_01_foundations_of_neural_networks/tests -q
uv run jupyter notebook classes/class_01_foundations_of_neural_networks/notebooks/flowers.ipynb
```
