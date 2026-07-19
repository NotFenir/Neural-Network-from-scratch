import numpy as np

from .LossBase import LossBase
from framework_types import Input


class L1Loss(LossBase):
    def loss(self, predicted: Input, target: Input) -> float:
        predicted, target = self.validate_data(predicted=predicted, target=target)
        return np.mean(np.abs(predicted - target))

    def gradient(self, predicted: Input, target: Input) -> np.ndarray:
        predicted, target = self.validate_data(predicted=predicted, target=target)
        return np.sign(predicted - target) / len(target)
