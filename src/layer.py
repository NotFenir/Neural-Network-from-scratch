from __future__ import annotations
import numpy as np
from typing import Callable

from framework_types import ModelPart


class Layer(ModelPart):
    """
    Docstring for Layer
    """

    def __init__(self, input_dim: int, output_dim: int, activation: str = None) -> None:
        self._weights: np.ndarray = np.random.randn(input_dim, output_dim) * np.sqrt(
            2 / input_dim
        )
        self._bias: np.ndarray = np.zeros(output_dim)
        self._activation: Callable = activation
        self._input_cache: np.ndarray = None

    def __call__(self, X: Layer | np.ndarray) -> np.ndarray:
        result = self.rmultiply_by(X)
        return result if self._activation is None else self._activation(result)

    def __matmul__(self, other: Layer | np.ndarray) -> np.ndarray:
        return self.multiply_by(other)

    def __rmatmul__(self, other: Layer | np.ndarray) -> np.ndarray:
        return self.rmultiply_by(other)

    def __repr__(self) -> str:
        return str(self._weights)

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @property
    def bias(self) -> np.ndarray:
        return self._bias

    @property
    def shape(self) -> tuple:
        return self._weights.shape

    @DeprecationWarning
    def multiply_by(self, other: Layer | np.ndarray) -> np.ndarray:
        raise NotImplementedError("Should not use this method!")
        if isinstance(other, np.ndarray):
            self._input_cache = other
            return self._weights @ other
        if isinstance(other, Layer):
            self._input_cache = other
            return self._weights @ other._weights + other._bias
        raise NotImplementedError(
            "Not implemented Layer's matmul for this specific case"
        )

    def rmultiply_by(self, other: Layer | np.ndarray) -> np.ndarray:
        if isinstance(other, np.ndarray):
            self._input_cache = other
            return (other @ self._weights) + self._bias
        if isinstance(other, Layer):
            self._input_cache = other
            return (other._weights @ self._weights) + self._bias
        raise NotImplementedError(
            "Not implemented Layer's rmatmul for this specific case"
        )

    def gradient_weights(self, dL_dz: np.ndarray | float) -> np.ndarray | float:
        print(f"dl: {type(dL_dz)} input: {self._input_cache}")
        return dL_dz * self._input_cache

    def gradient_bias(self, dL_dz: np.ndarray | float) -> np.ndarray | float:
        return dL_dz

    def calculate_gradient_weights(
        self, x: np.ndarray | float, dL_dz: np.ndarray | float
    ) -> np.ndarray | float:
        return dL_dz * x

    def calculate_gradient_bias(self, dL_dz: np.ndarray | float) -> np.ndarray | float:
        return dL_dz
