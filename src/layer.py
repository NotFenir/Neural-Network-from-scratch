from __future__ import annotations
import numpy as np
from typing import Callable


class Layer:
    """
    Docstring for Layer
    """

    def __init__(self, input_dim: int, output_dim: int, activation: str = None) -> None:
        self._weights: np.ndarray = np.random.randn(input_dim, output_dim) * np.sqrt(
            2 / input_dim
        )
        self._bias: np.ndarray = np.zeros(output_dim)
        self._activation: Callable = activation

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

    def multiply_by(self, other: Layer | np.ndarray) -> np.ndarray:
        if isinstance(other, np.ndarray):
            return self._weights @ other
        if isinstance(other, Layer):
            return self._weights @ other._weights + other._bias
        raise NotImplementedError(
            "Not implemented Layer's matmul for this specific case"
        )

    def rmultiply_by(self, other: Layer | np.ndarray) -> np.ndarray:
        if isinstance(other, np.ndarray):
            return (other @ self._weights) + self._bias
        if isinstance(other, Layer):
            return (other._weights @ self._weights) + self._bias
        raise NotImplementedError(
            "Not implemented Layer's rmatmul for this specific case"
        )
