from abc import ABC, abstractmethod
import numpy as np

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
        predicted = self.validate_data_type(predicted)
        target = self.validate_data_type(target)

        if not self.is_data_length_same(predicted=predicted, target=target):
            pred_size = (
                predicted.shape if hasattr(predicted, "shape") else len(predicted)
            )
            target_size = target.shape if hasattr(target, "shape") else len(target)

            raise ValueError(
                f"Mismatched data lanegths/shpes! Cannot calculate loss"
                f"Predicted size: {pred_size}, Target size: {target_size}"
            )

        return predicted, target

    def validate_data_type(self, x: LossInput):
        if not isinstance(x, ALLOWED_INPUT_TYPES):
            expected = ", ".join(t._name__ for t in ALLOWED_INPUT_TYPES)
            raise TypeError(f"Argument must be {expected}. Got {type(x).__name__}")

        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, float):
            return x
        if isinstance(x, (tuple, list)):
            return np.array(x)

    def is_data_length_same(self, predicted: LossInput, target: LossInput) -> bool:
        if not self.check_data_types(predicted, target):
            raise TypeError(
                f"Mismatched types! Both argument have to have the same type"
                f"Got predicted={type(predicted).__name__} and target={type(target).__name__}"
            )

        if isinstance(predicted, np.ndarray):
            return predicted.shape == target.shape
        if isinstance(predicted, float):
            return True
        if isinstance(predicted, (tuple, list)):
            return len(predicted) == len(target)

        raise NotImplementedError(
            f"Logic for type {type(predicted).__name__} is not implemented"
        )

    def check_data_types(self, predicted: LossInput, target: LossInput) -> bool:
        return type(predicted) is type(target)
