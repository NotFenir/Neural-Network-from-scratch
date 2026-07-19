from abc import ABC, abstractmethod
import numpy as np

from utils import validate_data
from framework_types import Input, ModelPart

ALLOWED_INPUT_TYPES = Input.__value__.__args__


class LossBase(ABC, ModelPart):
    def __call__(self, predicted: Input, target: Input) -> float:
        return self.loss(predicted=predicted, target=target)

    @abstractmethod
    def loss(self, predicted: Input, target: Input) -> float:
        pass

    @abstractmethod
    def gradient(self, predicted: Input, target: Input) -> np.ndarray:
        pass

    def validate_data(self, predicted: Input, target: Input) -> tuple[Input, Input]:
        return validate_data(predicted=predicted, target=target)
