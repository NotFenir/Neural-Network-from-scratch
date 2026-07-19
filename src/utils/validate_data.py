import numpy as np

from framework_types import Input

ALLOWED_INPUT_TYPES = Input.__value__.__args__


def validate_data(predicted: Input, target: Input) -> tuple[Input, Input]:
    predicted = validate_data_type(predicted)
    target = validate_data_type(target)

    if not is_data_length_same(predicted=predicted, target=target):
        pred_size = predicted.shape if hasattr(predicted, "shape") else len(predicted)
        target_size = target.shape if hasattr(target, "shape") else len(target)

        raise ValueError(
            f"Mismatched data lanegths/shpes! Cannot calculate loss"
            f"Predicted size: {pred_size}, Target size: {target_size}"
        )

    return predicted, target


def validate_data_type(x: Input):
    if not isinstance(x, ALLOWED_INPUT_TYPES):
        expected = ", ".join(t._name__ for t in ALLOWED_INPUT_TYPES)
        raise TypeError(f"Argument must be {expected}. Got {type(x).__name__}")

    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, float):
        return x
    if isinstance(x, (tuple, list)):
        return np.array(x)


def check_data_types(predicted: Input, target: Input) -> bool:
    return type(predicted) is type(target)


def is_data_length_same(predicted: Input, target: Input) -> bool:
    if not check_data_types(predicted, target):
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
