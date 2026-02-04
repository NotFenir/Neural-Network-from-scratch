import numpy as np

class Model:
    '''
    Docstring for Model
    '''
    
    
    def __init__(self, *args, **kwargs):
        ...
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x)
    
    
    def fit(self) -> None:
        ...
        
    def predict(self, x: np.ndarray) -> np.ndarray:
        ...
        