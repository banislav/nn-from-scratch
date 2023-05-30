from typing import List


class Layer:
    def __init__(self):
        pass

    def forward(self, input_data: List):
        return input_data

    def backward(self, grads: List, input_data: List):
        pass
