from .LossBase import LossBase, LossInput
import numpy as np


class L2Loss(LossBase):
    def __call__(self, predicted: LossInput, target: LossInput) -> float:
        return self.loss(predicted=predicted, target=target)

    def loss(self, predicted: LossInput, target: LossInput) -> float:
        return 0.5 * (predicted - target) ** 2

    def gradient(self, predicted: LossInput, target: LossInput) -> np.ndarray:
        return predicted - target
