from abc import ABC, abstractmethod
import numpy as np

from utils import validate_data

type LossInput = np.ndarray | float | list | tuple
type LossOutput = float

ALLOWED_INPUT_TYPES = LossInput.__value__.__args__


class LossBase(ABC):
    def __call__(self, predicted: LossInput, target: LossInput) -> float:
        return self.loss(predicted=predicted, target=target)

    @abstractmethod
    def loss(self, predicted: LossInput, target: LossInput) -> float:
        pass

    @abstractmethod
    def gradient(self, predicted: LossInput, target: LossInput) -> np.ndarray:
        pass

    def validate_data(
        self, predicted: LossInput, target: LossInput
    ) -> tuple[LossInput, LossInput]:
        return validate_data(predicted=predicted, target=target)
