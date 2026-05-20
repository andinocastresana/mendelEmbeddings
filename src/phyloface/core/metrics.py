# =========================================
# ID: PHYLOFACE_CORE_BASE_001
# VERSION: v1.0
# =========================================
# Origen de las funciones (migración Tarea #1, Paso 6a):
#   src/phyloface_experimental_functions.py
#   - l2_normalize          (línea 392, NUEVA)
#   - cosine_similarity     (línea 406, REEMPLAZA la versión previa)
#   - cosine_distance       (línea 418, NUEVA)
#   - euclidean_distance    (línea 428, REEMPLAZA la versión previa)
#
# Esta entrega cubre 2 archivos (mismo ID):
#   1) phyloface/core/io.py        (ver allí)
#   2) phyloface/core/metrics.py   (este archivo)
#
# === Cambio de comportamiento respecto a la versión previa ===
# La versión anterior de este archivo (sin ID, parte del bundle original
# `PHYLOFACE_BACKEND_001`) tenía dos implementaciones distintas:
#
#   * `cosine_similarity`: dividía v por norm(v) inline, sin manejar
#     norm=0 ni forzar dtype. Matemáticamente equivalente a la nueva
#     para vectores no-cero; la diferencia es defensiva.
#
#   * `euclidean_distance`: calculaba `norm(v1 - v2)` SIN NORMALIZAR
#     los vectores antes. Para embeddings de ArcFace (norma ~22),
#     devolvía valores grandes proporcionales a las magnitudes.
#     La nueva versión normaliza primero, devolviendo valores en [0, 2].
#
# **Decisión 2026-05-20**: reemplazar ambas con la versión experimental
# que usa `l2_normalize`, porque es internamente consistente entre
# coseno y euclídea (ambas trabajan sobre vectores unitarios) y es el
# estándar en literatura de face recognition. Consecuencia: cualquier
# umbral o resultado numérico generado previamente con la `euclidean_distance`
# antigua deja de ser comparable. No había umbrales calibrados todavía
# (Tarea #6 del proyecto los va a generar contra KinFaceW), así que el
# impacto es chico. Documentado en `_meta/MIGRACION_TAREA1.md`.
#
# `get_metric_function` y `get_metric_label` se preservan tal cual: son
# helpers de dispatch que no cambian con el reemplazo interno.

# -----------------------------------------
# FILE: phyloface/core/metrics.py
# -----------------------------------------

import numpy as np


# =========================================================
# 1) Normalización L2: vector -> vector unitario
# =========================================================
# Pasa un vector a norma L2 = 1. Es la base para que coseno y euclídea
# trabajen sobre vectores unitarios y por tanto sean métricas comparables
# entre sí en magnitud.
# Defensivo en 3 frentes:
#   - asarray + float32: tolera listas, tuplas, ndarrays de otros dtypes.
#   - ravel: tolera shapes raras (matriz fila, columna, etc.).
#   - norm=0: devuelve el vector tal cual (no divide). Esto puede pasar
#     con embeddings degenerados o sintéticos.
# No depende de funciones propias del módulo.
def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """
    Normaliza un vector a norma L2 = 1.

    Parámetros:
        vec: vector (cualquier shape; se aplana con ravel).

    Devuelve:
        ndarray float32 unidimensional. Si la norma de entrada es 0,
        devuelve el vector (aplanado) tal cual.
    """
    vec = np.asarray(vec, dtype=np.float32).ravel()
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


# =========================================================
# 2) Similitud coseno entre dos vectores
# =========================================================
# Calcula coseno entre dos vectores, normalizándolos primero con
# `l2_normalize`. Resultado en [-1, 1].
#   +1 → vectores idénticamente orientados (máximo parecido).
#    0 → ortogonales (sin correlación).
#   -1 → opuestos.
# Para embeddings de ArcFace (típicamente >= 0 entre caras humanas),
# los valores prácticos suelen estar en [0.0, 1.0].
# Depende de: l2_normalize.
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Similitud coseno.
    Mayor valor = mayor parecido.
    """
    v1 = l2_normalize(vec1)
    v2 = l2_normalize(vec2)
    return float(np.dot(v1, v2))


# =========================================================
# 3) Distancia coseno = 1 - similitud coseno
# =========================================================
# Misma información que cosine_similarity, expresada como distancia
# (rango [0, 2]: 0 = idénticos, 2 = opuestos). Se incluye porque a
# veces es más cómodo trabajar con distancias que con similitudes
# (ej. para clustering o thresholding por debajo de un umbral).
# Depende de: cosine_similarity.
def cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Distancia coseno (= 1 - similitud coseno).
    Menor valor = mayor parecido.
    """
    return 1.0 - cosine_similarity(vec1, vec2)


# =========================================================
# 4) Distancia euclídea (sobre vectores normalizados)
# =========================================================
# Calcula la distancia euclídea entre dos vectores **después de
# normalizarlos a norma unitaria**. Para vectores unitarios, el rango
# de la distancia euclídea es [0, 2]:
#   - 0     → vectores idénticamente orientados.
#   - sqrt(2) ≈ 1.414 → ortogonales.
#   - 2     → opuestos.
# Esta relación con coseno es: eucl² = 2 (1 - cos), o sea, no aporta
# información ortogonal pero responde distinto a outliers.
# Depende de: l2_normalize.
def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Distancia euclídea entre vectores L2-normalizados.
    Menor valor = mayor parecido.
    """
    v1 = l2_normalize(vec1)
    v2 = l2_normalize(vec2)
    return float(np.linalg.norm(v1 - v2))


# =========================================================
# 5) Dispatch por nombre (helper para configurar pipelines)
# =========================================================
# Permite seleccionar una métrica a partir de un string (ej. desde
# config o CLI). Mantiene API estable: agregar una métrica nueva
# requiere actualizar SOLO esta función + sus consumidores que
# pasen por `metric_name`.
# No depende de funciones propias del módulo.
def get_metric_function(metric_name: str):
    """
    Devuelve la función de comparación según el nombre configurado.

    Soportadas: 'cosine' (→ cosine_similarity) y 'euclidean'
    (→ euclidean_distance).
    """
    metric_name = metric_name.lower()

    if metric_name == "cosine":
        return cosine_similarity

    if metric_name == "euclidean":
        return euclidean_distance

    raise ValueError(
        f"Métrica no soportada: {metric_name}. Usa 'cosine' o 'euclidean'."
    )


# =========================================================
# 6) Etiqueta humana de la métrica (para títulos/colorbars)
# =========================================================
# Devuelve un texto presentable para usar en plots (título, colorbar).
# Si el nombre no es reconocido, devuelve el nombre tal cual.
# No depende de funciones propias del módulo.
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
