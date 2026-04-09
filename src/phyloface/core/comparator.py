# -----------------------------------------
# FILE: phyloface/core/comparator.py
# -----------------------------------------
import numpy as np

from phyloface.core.models import DetectedFace
from phyloface.core.metrics import get_metric_function


class FaceComparator:
    """
    Compara listas de rostros mediante embeddings.
    """

    def compare_sets(
        self,
        faces_a: list[DetectedFace],
        faces_b: list[DetectedFace],
        metric: str = "cosine",
    ) -> np.ndarray:
        """
        Devuelve una matriz de score [len(faces_a), len(faces_b)].
        """
        score_fn = get_metric_function(metric)
        score_matrix = np.zeros((len(faces_a), len(faces_b)), dtype=float)

        for i, face_a in enumerate(faces_a):
            for j, face_b in enumerate(faces_b):
                score_matrix[i, j] = score_fn(
                    face_a.embedding,
                    face_b.embedding,
                )

        return score_matrix
