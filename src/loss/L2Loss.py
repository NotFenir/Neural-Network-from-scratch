import numpy as np

from .LossBase import LossBase
from framework_types import Input


class L2Loss(LossBase):
    def loss(self, predicted: Input, target: Input) -> float:
        predicted, target = self.validate_data(predicted=predicted, target=target)
        return np.mean(0.5 * (predicted - target) ** 2)

    def gradient(self, predicted: Input, target: Input) -> np.ndarray:
        predicted, target = self.validate_data(predicted=predicted, target=target)
        return predicted - target
