import numpy as np

from framework_types import ModelSequence
from layer import Layer
from activations import ActivationFuncBase


class Model:
    """
    Docstring for Model
    """

    def __init__(self, sequence: ModelSequence):
        self._sequence: ModelSequence = sequence
        self._has_forwarded: bool = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int, optimizer) -> None:
        pass

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._has_forwarded = True
        x = self._sequence[0](X)

        for layer in self._sequence[1:]:
            x = layer(x)
        return x

    def backward(self, loss_value: float) -> None:
        if not self._has_forwarded:
            raise ValueError("There was no forward passed")

        delta: np.ndarray | float = loss_value

        for layer in self._sequence[::-1]:
            ...
            # if isinstance(layer, Layer):
            #     print(layer.gradient_weights(delta))
            # elif isinstance(layer, ActivationFuncBase):
            #     print(layer.gradient())
