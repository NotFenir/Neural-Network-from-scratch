import numpy as np

class Layer:
    '''
    Docstring for Layer
    '''
    
    def __init__(self, input_dim: int, output_dim: int) -> None:
        self._weights: np.ndarray = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
        self._bias: np.ndarray = np.zeros(output_dim)
        
    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X @ self._weights) #+ self.bias
        
    @property
    def weights(self):
        return self._weights
    
    @property
    def bias(self):
        return self._bias