from .LossBase import LossBase, LossInput
import numpy as np


class L1Loss(LossBase):
    def loss(self, predicted: LossInput, target: LossInput) -> float:
        predicted, target = self.validate_data(predicted=predicted, target=target)
        return np.mean(np.abs(predicted - target))

    def gradient(self, predicted: LossInput, target: LossInput) -> np.ndarray:
        predicted, target = self.validate_data(predicted=predicted, target=target)
        return np.sign(predicted - target) / len(target)
