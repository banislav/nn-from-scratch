import numpy as np
from layer import Layer
from typing import List


class DenseLayer(Layer):
    def __init__(self, n_inputs, n_neurons, learning_rate=0.01):
        super().__init__()
        self.learning_rate = learning_rate
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons

    def forward(self, input_data: List):
        pass

    def backward(self, grads, input_data):
        pass