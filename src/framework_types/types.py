import numpy as np

from .class_types import ModelPart

type Input = np.ndarray | float | int | list | tuple | np.float64
type ModelSequence = list[ModelPart] | tuple[ModelPart]
