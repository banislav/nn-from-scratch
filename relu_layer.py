import numpy as np

from layer import Layer


class ReluLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_data: np.ndarray):
        return np.maximum(0, input_data)

    def backward(self, grads: np.ndarray, X: np.ndarray):
        return grads * (X > 0)
