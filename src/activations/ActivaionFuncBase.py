from abc import ABC, abstractmethod
import numpy as np

from utils import validate_data_type
from framework_types import Input


ALLOWED_INPUT_TYPES = Input.__value__.__args__


class ActivationFuncBase(ABC):
    def __call__(self, x: Input) -> np.ndarray | np.float64:
        return self.calculate(x)

    @abstractmethod
    def calculate(self, x: Input) -> np.ndarray | np.float64:
        pass

    @abstractmethod
    def gradient(self, x: Input) -> np.ndarray | np.float64:
        pass

    def validate_data(self, x: Input):
        return validate_data_type(x)
