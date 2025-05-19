from typing import Tuple
import numpy as np
from PIL import Image

class Resize:
    def __init__(self, size: Tuple) -> None:
        self.size = size
    
    def transform(self, image: np.ndarray) -> np.ndarray:
        pil_image = Image.fromarray(image.astype(np.uint8),mode='RGB')
        pil_image = pil_image.resize(self.size)
        image = np.array(pil_image)
        return image