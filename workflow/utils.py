
import numpy as np
import cv2 as cv
from PIL import Image

def pil_to_cv2_bgr(pil_image: Image.Image) -> np.ndarray:
    """Converts a PIL Image (RGB) to an OpenCV BGR NumPy array."""
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    return np.array(pil_image)[:, :, ::-1]

def cv2_bgr_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """Converts an OpenCV BGR NumPy array to a PIL Image (RGB)."""
    rgb_image = cv.cvtColor(cv2_image, cv.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)

def bgr_to_lab(bgr_image: np.ndarray) -> np.ndarray:
    """Converts a BGR NumPy array to a LAB NumPy array."""
    return cv.cvtColor(bgr_image, cv.COLOR_BGR2LAB)

def lab_to_bgr(lab_image: np.ndarray) -> np.ndarray:
    """Converts a LAB NumPy array to a BGR NumPy array."""
    return cv.cvtColor(lab_image, cv.COLOR_LAB2BGR)
