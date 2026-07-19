from abc import ABC, abstractmethod

import numpy as np

from framework_types import Input, ModelPart
from utils import validate_data_type

ALLOWED_INPUT_TYPES = Input.__value__.__args__


class ActivationFuncBase(ABC, ModelPart):
    def __call__(self, x: Input) -> np.ndarray | np.float64:
        return self.calculate(x)

    @abstractmethod
    def calculate(self, x: Input) -> np.ndarray | np.float64:
        pass

    @abstractmethod
    def gradient(self) -> np.ndarray | float:
        pass

    @abstractmethod
    def calculate_gradient(self, x: Input) -> np.ndarray | np.float64:
        pass

    def validate_data(self, x: Input):
        return validate_data_type(x)
