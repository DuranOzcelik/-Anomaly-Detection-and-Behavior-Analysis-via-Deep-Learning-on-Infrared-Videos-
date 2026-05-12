import cv2
import numpy as np
from typing import Tuple

def load_thermal_image(image_path: str) -> np.ndarray:
    """Thermal image yükleme"""
    img = cv2.imread(image_path)
    return img

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Görüntü ön işleme"""
    # Normalize
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized

def detect_anomalies(image: np.ndarray) -> Tuple[bool, float]:
    """Anomali tespiti"""
    # Placeholder
    return False, 0.0
