import numpy as np
from .ActivaionFuncBase import ActivationFuncBase
from framework_types import Input


class Sigmoid(ActivationFuncBase):
    def calculate(self, x: Input) -> np.ndarray | np.float64:
        x = self.validate_data(x)
        return self._eval(x)

    def gradient(self, x: Input) -> np.ndarray | np.float64:
        x = self.validate_data(x)
        return self._eval(x) * (1 - self._eval(x))

    @classmethod
    def _eval(cls, x: Input) -> np.ndarray | np.float64:
        return 1 / (1 + np.exp(-x))
