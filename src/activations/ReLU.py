import numpy as np
from .ActivaionFuncBase import ActivationFuncBase
from framework_types import Input


class ReLU(ActivationFuncBase):
    def calculate(self, x: Input) -> np.ndarray | np.float64:
        x = self.validate_data(x)
        return np.max((0, x))

    def gradient(self, x):
        x = self.validate_data(x)
        return 1 if x >= 0 else 0
