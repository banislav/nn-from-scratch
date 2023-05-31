import numpy as np


class Layer:
    def __init__(self):
        pass

    def forward(self, X: np.ndarray):
        return X

    def backward(self, grads: np.ndarray, X: np.ndarray):
        pass
