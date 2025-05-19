from typing import Tuple
import numpy as np

class Normalize:
    def __init__(self, mean: Tuple | float, std: Tuple | float) -> None:
        self.mean = mean if isinstance(mean, (tuple, list)) else (mean,)
        self.std = std if isinstance(std, (tuple, list)) else (std,)

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        shape = matrix.shape
        matrix = matrix.astype(np.float32) / 255.0 

        if len(shape) == 3 and shape[2] in [1, 3]:  
            matrix = matrix.transpose(2, 0, 1)  
            for i in range(matrix.shape[0]):
                matrix[i] = (matrix[i] - self.mean[i % len(self.mean)]) / self.std[i % len(self.std)]
            matrix = matrix.transpose(1, 2, 0)  
        elif len(shape) == 2: 
            matrix = (matrix - self.mean[0]) / self.std[0]
        else:
            raise ValueError(f"Unsupported matrix shape: {shape}")

        assert matrix.shape == shape
        return matrix