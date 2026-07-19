from .LossBase import LossBase, LossInput
import numpy as np


class L1Loss(LossBase):
    def __call__(self, predicted: LossInput, target: LossInput) -> float:
        return self.loss(predicted=predicted, target=target)

    def loss(self, predicted: LossInput, target: LossInput) -> float:
        return np.mean(np.abs(predicted - target))

    def gradient(self, predicted: LossInput, target: LossInput) -> np.ndarray:
        return np.sign(predicted - target) / len(target)
