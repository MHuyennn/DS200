from typing import List
import numpy as np

class Transforms:
    def __init__(self, transforms : List) -> None:
        self.transforms = transforms

    def transform(self, matrix : np.ndarray) -> np.ndarray:
        for transformation in self.transforms:
            matrix = transformation.transform(matrix)
        return matrix