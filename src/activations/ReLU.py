import numpy as np

from framework_types import Input

from .ActivationFuncBase import ActivationFuncBase


class ReLU(ActivationFuncBase):
    def __init__(self):
        self._input_cache: np.ndarray = None

    def calculate(self, x: Input) -> np.ndarray | float:
        x = self.validate_data(x)
        self._input_cache = x
        return np.maximum(0, x)

    def calculate_gradient(self, x: Input) -> np.ndarray | float:
        x = self.validate_data(x)
        return 1 if x >= 0 else 0

    def gradient(self) -> np.ndarray | float:
        return np.where(self._input_cache >= 0, 1, 0)
