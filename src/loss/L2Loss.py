from .LossBase import LossBase, LossInput
import numpy as np


class L2Loss(LossBase):
    def loss(self, predicted: LossInput, target: LossInput) -> float:
        predicted, target = self.validate_data(predicted=predicted, target=target)
        return np.mean(0.5 * (predicted - target) ** 2)

    def gradient(self, predicted: LossInput, target: LossInput) -> np.ndarray:
        predicted, target = self.validate_data(predicted=predicted, target=target)
        return predicted - target
