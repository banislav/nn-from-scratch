import numpy as np
from layer import Layer
from typing import List


class DenseLayer(Layer):
    def __init__(self, n_inputs: int, n_outputs: int, learning_rate: int = 0.01):
        super().__init__()
        self.learning_rate = learning_rate
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.weights = np.random.normal(loc=0.0,
                                        scale=np.sqrt(2 / (n_inputs + n_outputs)),
                                        size=(n_inputs, n_outputs))
        self.biases = np.zeros_like(n_outputs)

    def forward(self, X: List[List, List]) -> List[List, List]:
        return np.dot(X, self.weights) + self.biases

    def backward(self, X: np.array, grads: np.array):
        loss_grads = np.dot(self.weights, grads)

        self.weights -= np.dot(X, grads) * self.learning_rate
        self.biases -= self.learning_rate * grads.mean(axis=0)*X.shape[0]

        return loss_grads
