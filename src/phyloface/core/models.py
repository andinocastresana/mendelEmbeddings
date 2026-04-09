# =========================================
# FILE: phyloface/core/models.py
# =========================================
from dataclasses import dataclass
import numpy as np

@dataclass
class DetectedFace:
    """
    Representa un rostro detectado y preparado para comparación.
    """
    index: int
    bbox: tuple[int, int, int, int]
    crop: np.ndarray
    embedding: np.ndarray
