import numpy as np

from framework_types import Input

from .ActivationFuncBase import ActivationFuncBase


class Sigmoid(ActivationFuncBase):
    def __init__(self) -> None:
        self._input_cache: np.ndarray = None

    def calculate(self, x: Input) -> np.ndarray | float:
        x = self.validate_data(x)
        self._input_cache = x
        return self._eval(x)

    def calculate_gradient(self, x: Input) -> np.ndarray | float:
        x = self.validate_data(x)
        return self._eval(x) * (1 - self._eval(x))

    def gradient(self) -> np.ndarray | float:
        return self._eval(self._input_cache) * (1 - self._eval(self._input_cache))

    @classmethod
    def _eval(cls, x: Input) -> np.ndarray | np.float64:
        return 1 / (1 + np.exp(-x))
