import numpy as np

from layer import Layer


class DenseLayer(Layer):
    def __init__(self, n_inputs: int, n_outputs: int, learning_rate: int = 0.01):
        super().__init__()
        self.learning_rate = learning_rate
        self.weights = np.random.normal(loc=0.0,
                                        scale=np.sqrt(2 / (n_inputs + n_outputs)),
                                        size=(n_inputs, n_outputs))
        self.biases = np.zeros((n_outputs,), dtype=np.float64)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.biases

    def backward(self, X: np.ndarray, grads: np.ndarray):
        self.weights -= np.dot(X.T, grads) * self.learning_rate
        self.biases -= self.learning_rate * np.array(grads.mean(axis=0) * X.shape[0]).T

        return np.dot(grads, self.weights.T)