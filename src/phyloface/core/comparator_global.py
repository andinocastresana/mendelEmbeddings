# =========================================
# ID: PHYLOFACE_COMPARATOR_GLOBAL_001
# VERSION: v1.0
# =========================================
# Origen de las funciones (migración Tarea #1, Paso 6d):
#   src/phyloface_experimental_functions.py
#   - compute_global_metrics   (línea 454)
#   - print_global_summary     (línea 940)
#
# Qué hace este módulo:
# - Es el **integrador del camino global** del par. Combina los 3 sub-pasos
#   anteriores del Paso 6:
#     * 6a (`core.metrics`)  → similitud coseno, distancia coseno, euclídea.
#     * 6b (`core.embedder`) → submodelo de reconocimiento + re-extracción
#                              de embeddings sobre la cara alineada.
#     * 6c (`core.pairs`)    → estructura `selected_pair` con face_a/face_b
#                              y aligned_a/aligned_b.
#   Y produce dos sub-diccionarios que se anidan dentro del `selected_pair`:
#     - `embedding_qc`   : métricas de control de calidad del re-embedding
#                          (cuán parecido es el embedding antes vs después
#                          de alinear, en cada cara y entre pares).
#     - `global_scores`  : las métricas principales del par — coseno y
#                          euclídea, antes y después de alinear.
#
# - `print_global_summary` imprime una versión humano-amigable del par
#   (face_id, scores principales y QC). Se usa típicamente desde notebook
#   o CLI para validar rápidamente que el flujo se ejecutó.
#
# Por qué guardar también métricas "_original":
# - Antes de alinear, el embedding ya viene calculado por el detector
#   (sobre el crop con alineación interna de InsightFace de 5-puntos).
# - Después de re-alinear con nuestro pipeline (más margen, tamaño
#   controlado), el embedding cambia. La diferencia (`pair_similarity_delta`,
#   `self_similarity_*_original_vs_post_align`) es un control de calidad:
#   - Si la self-similarity es muy baja (<<0.95) hay riesgo de que la
#     re-alineación esté distorsionando la cara y el embedding sea ruidoso.
#   - Si el delta entre embeddings originales y re-alineados es grande,
#     conviene revisar parámetros de alineación.
#
# **Cambio funcional heredado del Paso 6a**: las claves
# `euclidean_distance_original` y `euclidean_distance_post_align` ahora
# devuelven valores en `[0, 2]` (vectores L2-normalizados antes de la
# resta), no valores grandes proporcionales a la magnitud del embedding
# como pasaba en la implementación previa del paquete. Ver
# `_meta/MIGRACION_TAREA1.md` para el contexto completo de esa decisión.
#
# Cosas que NO hace este módulo:
# - No calcula scores regionales (eso es `comparator_regional.py`).
# - No grafica nada (eso es `viz/`).
# - No persiste el selected_pair en disco (sin caché propio).

# -----------------------------------------
# FILE: phyloface/core/comparator_global.py
# -----------------------------------------

import numpy as np

from phyloface.core.embedder import (
    get_recognition_model,
    extract_embedding_from_aligned,
)
from phyloface.core.metrics import (
    cosine_similarity,
    cosine_distance,
    euclidean_distance,
)


# =========================================================
# 1) Cálculo integrador: embeddings post-align + QC + scores globales
# =========================================================
# Toma el `selected_pair` (con aligned_a/aligned_b ya calculados por
# `build_selected_pair`) y el `FaceAnalysis` (para acceder al submodelo
# de reconocimiento). Calcula y guarda en el mismo dict:
#   - embedding_a/b_original     : ndarrays del detector inicial (raveled).
#   - embedding_a/b_post_align   : ndarrays re-extraídos sobre los crops
#                                  alineados por nuestro pipeline.
#   - embedding_qc               : 5 métricas de control de calidad.
#   - global_scores              : 6 métricas principales (coseno y
#                                  euclídea, original y post-align).
#
# Devuelve el mismo `selected_pair` mutado (in-place + return).
#
# Depende de: get_recognition_model, extract_embedding_from_aligned,
#             cosine_similarity, cosine_distance, euclidean_distance.
def compute_global_metrics(app, selected_pair: dict) -> dict:
    """
    Calcula embeddings post-align, QC y métricas globales del par.

    Parámetros:
        app: instancia de FaceAnalysis (ver `core.pairs.init_face_app`).
        selected_pair: dict producido por `core.pairs.build_selected_pair`.
            Debe contener: 'face_a', 'face_b', 'aligned_a', 'aligned_b'.

    Devuelve:
        El mismo `selected_pair`, ahora con 6 claves nuevas:
            'embedding_a_original', 'embedding_b_original',
            'embedding_a_post_align', 'embedding_b_post_align',
            'embedding_qc', 'global_scores'.
    """
    # Submodelo de reconocimiento (descubre el primero con `get_feat`).
    rec_model = get_recognition_model(app)

    # Cara A, cara B, recortes alineados (los pone build_selected_pair).
    face_a = selected_pair["face_a"]
    face_b = selected_pair["face_b"]
    aligned_a = selected_pair["aligned_a"]
    aligned_b = selected_pair["aligned_b"]

    # ---- Embeddings ORIGINALES (del detector, sobre crop sin re-alinear) ----
    # `ravel()` defensivo por si el embedding del detector vino como (1, D).
    # `asarray + dtype=float32` por si vino con otro dtype.
    emb_a_original = np.asarray(face_a["embedding"], dtype=np.float32).ravel()
    emb_b_original = np.asarray(face_b["embedding"], dtype=np.float32).ravel()

    # ---- Embeddings POST-ALIGN (re-extraídos sobre las caras alineadas) ----
    # Son los que mejor reflejan el rostro tal como lo procesa nuestro
    # pipeline (con margen y tamaño controlados).
    emb_a_post = extract_embedding_from_aligned(rec_model, aligned_a)
    emb_b_post = extract_embedding_from_aligned(rec_model, aligned_b)

    # ---- QC: cuán parecidos son los embeddings de cada cara antes y
    # después de alinear, y cómo cambió la similitud del par. ----
    qc = {
        # Self-similarity: idealmente >= 0.95. Bajos valores indican que
        # la re-alineación distorsionó la cara significativamente.
        "self_similarity_a_original_vs_post_align": cosine_similarity(emb_a_original, emb_a_post),
        "self_similarity_b_original_vs_post_align": cosine_similarity(emb_b_original, emb_b_post),
        # Pair-similarity antes y después de re-alinear: los guardamos
        # ambos para poder calcular el delta abajo.
        "pair_similarity_original": cosine_similarity(emb_a_original, emb_b_original),
        "pair_similarity_post_align": cosine_similarity(emb_a_post, emb_b_post),
    }
    # Delta = ganancia (positivo) o pérdida (negativo) de similitud al
    # re-alinear. Positivos sugieren que la re-alineación mejoró la
    # comparabilidad; negativos sugieren que introdujo ruido.
    qc["pair_similarity_delta"] = qc["pair_similarity_post_align"] - qc["pair_similarity_original"]

    # ---- Scores GLOBALES principales: 6 métricas, dos versiones de cada par. ----
    # Decisión: las tres métricas (coseno, distancia coseno, euclídea)
    # se calculan tanto sobre embeddings originales como post-align,
    # para que el usuario pueda comparar las dos vías. Los scores que
    # se usan típicamente para reportar son los `_post_align`.
    scores = {
        "cosine_similarity_original":     cosine_similarity(emb_a_original, emb_b_original),
        "cosine_similarity_post_align":   cosine_similarity(emb_a_post,     emb_b_post),
        "cosine_distance_original":       cosine_distance(emb_a_original,   emb_b_original),
        "cosine_distance_post_align":     cosine_distance(emb_a_post,       emb_b_post),
        # Atención: estas euclídeas vienen en [0, 2] por la nueva
        # semántica de `core.metrics.euclidean_distance` (Paso 6a).
        "euclidean_distance_original":    euclidean_distance(emb_a_original, emb_b_original),
        "euclidean_distance_post_align":  euclidean_distance(emb_a_post,     emb_b_post),
    }

    # ---- Mutación in-place del selected_pair ----
    # Guardamos los 4 embeddings, el dict QC y el dict de scores.
    selected_pair["embedding_a_original"] = emb_a_original
    selected_pair["embedding_b_original"] = emb_b_original
    selected_pair["embedding_a_post_align"] = emb_a_post
    selected_pair["embedding_b_post_align"] = emb_b_post
    selected_pair["embedding_qc"] = qc
    selected_pair["global_scores"] = scores

    return selected_pair


# =========================================================
# 2) Print humano-amigable de los scores globales y del QC
# =========================================================
# Imprime un resumen del `selected_pair` tras pasar por `compute_global_metrics`:
#   - face_ids de A y B (para identificar el par).
#   - 3 scores principales post-align (los que típicamente se reportan).
#   - 3 métricas de QC (self-sims + delta).
# Está pensada para usarse desde notebook tras la corrida — formato
# legible, no estructurado.
# No depende de funciones propias del módulo.
def print_global_summary(selected_pair: dict):
    """
    Resumen corto y legible de la comparación global de un par.
    """
    # IDs y sub-dicts a imprimir.
    face_a = selected_pair["face_a"]["face_id"]
    face_b = selected_pair["face_b"]["face_id"]
    scores = selected_pair["global_scores"]
    qc = selected_pair["embedding_qc"]

    # Bloque 1: scores principales (los post-align son los que más importan).
    print("=== COMPARACIÓN GLOBAL ===")
    print(f"{face_a} vs {face_b}")
    print(f"cosine_similarity_post_align : {scores['cosine_similarity_post_align']:.4f}")
    print(f"cosine_distance_post_align   : {scores['cosine_distance_post_align']:.4f}")
    print(f"euclidean_distance_post_align: {scores['euclidean_distance_post_align']:.4f}")
    print()

    # Bloque 2: QC del embedding (self-sim de cada cara + delta del par).
    # Se imprime el `+.4f` para el delta para dejar explícito el signo.
    print("=== QC EMBEDDING ===")
    print(f"self_similarity_a_original_vs_post_align: {qc['self_similarity_a_original_vs_post_align']:.4f}")
    print(f"self_similarity_b_original_vs_post_align: {qc['self_similarity_b_original_vs_post_align']:.4f}")
    print(f"pair_similarity_delta                  : {qc['pair_similarity_delta']:+.4f}")
