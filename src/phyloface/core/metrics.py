# -----------------------------------------
# FILE: phyloface/core/metrics.py
# -----------------------------------------
import numpy as np


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Similitud coseno.
    Mayor valor = mayor parecido.
    """
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    return float(np.dot(vec1, vec2))


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Distancia euclídea.
    Menor valor = mayor parecido.
    """
    return float(np.linalg.norm(vec1 - vec2))


def get_metric_function(metric_name: str):
    """
    Devuelve la función de comparación según el nombre configurado.
    """
    metric_name = metric_name.lower()

    if metric_name == "cosine":
        return cosine_similarity

    if metric_name == "euclidean":
        return euclidean_distance

    raise ValueError(
        f"Métrica no soportada: {metric_name}. Usa 'cosine' o 'euclidean'."
    )


def get_metric_label(metric_name: str) -> str:
    """
    Texto para usar en el colorbar / títulos.
    """
    metric_name = metric_name.lower()

    if metric_name == "cosine":
        return "Cosine similarity"

    if metric_name == "euclidean":
        return "Euclidean distance"

    return metric_name

