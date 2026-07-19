from abc import ABC, abstractmethod
import numpy as np

type LossInput = np.ndarray | float
type LossOutput = float


class LossBase(ABC):
    @abstractmethod
    def loss(self, predicted: LossInput, target: LossInput) -> float:
        pass

    @abstractmethod
    def gradient(self, predicted: LossInput, target: LossInput) -> np.ndarray:
        pass
