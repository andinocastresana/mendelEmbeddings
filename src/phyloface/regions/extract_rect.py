# =========================================
# ID: PHYLOFACE_REGIONS_001
# VERSION: v1.0
# =========================================
# Origen de las funciones (migración Tarea #1):
#   src/phyloface_experimental_functions.py
#   - extract_regions_v2          (línea 745)
#   - add_regions_v2_to_pair      (línea 816)
#
# Qué hace este módulo:
# - Orquesta la extracción de regiones faciales **rectangulares** a partir
#   de landmarks densos. Es la versión "regiones v2" del proyecto: bbox +
#   crop RGB para cada región anatómica, sin máscara poligonal.
# - Define 12 regiones: ojos (izq/der), boca, nariz, cejas (izq/der),
#   pómulos (izq/der), mejillas (izq/der), mentón y frente. Las primeras
#   tres están marcadas como `source="official"` (vienen de Face Mesh);
#   las demás como `source="approx"`.
# - La función `add_regions_v2_to_pair` aplica la extracción a las dos
#   caras alineadas de un `selected_pair` y guarda el resultado bajo
#   `regions_v2 = {"A": ..., "B": ...}`.
#
# Decisiones de diseño:
# - El padding por región (0.25 ojos/boca, 0.22 nariz, 0.20 resto) se deja
#   tal cual el archivo original. Si más adelante se quiere parametrizar,
#   conviene hacerlo a través de un dict externo, no a fuerza de booleans.
# - Cada región devuelve un dict con la misma estructura:
#     { "bbox": (x1,y1,x2,y2), "crop_rgb": ndarray, "landmark_idx": list|None,
#       "source": "official" | "approx" }
#   Esta estructura es contrato: `extract_masked.py` la enriquece luego
#   con "mask" y "crop_masked_rgb", y `viz/regions.py` la consume tal cual.
# - `forehead` queda fuera del loop principal porque su bbox se calcula
#   con `get_forehead_bbox` (no con `get_region_bbox`); por eso se agrega
#   al final, con `landmark_idx=None`.
#
# Cosas que NO hace este módulo:
# - No extrae máscaras poligonales (eso es `extract_masked.py`, paso siguiente).
# - No calcula métricas. Solo recorta y arma el dict de regiones.

# -----------------------------------------
# FILE: phyloface/regions/extract_rect.py
# -----------------------------------------

import numpy as np

from phyloface.regions.geometry import (
    # Constantes con los índices de cada región
    LEFT_EYE_IDX,
    RIGHT_EYE_IDX,
    LIPS_IDX,
    NOSE_IDX,
    LEFT_EYEBROW_IDX,
    RIGHT_EYEBROW_IDX,
    LEFT_CHEEKBONE_IDX,
    RIGHT_CHEEKBONE_IDX,
    LEFT_CHEEK_IDX,
    RIGHT_CHEEK_IDX,
    CHIN_IDX,
    # Helpers geométricos
    get_region_bbox,
    crop_from_bbox,
    get_forehead_bbox,
    get_chin_bbox_refined,
)


# =========================================================
# 1) Extracción de regiones (cara única)
# =========================================================
# Recibe una imagen RGB ya alineada y sus landmarks densos.
# Devuelve un dict { region_name: {bbox, crop_rgb, landmark_idx, source} }
# con las 12 regiones del pipeline.
#
# Depende de: get_chin_bbox_refined, get_region_bbox, crop_from_bbox,
#             get_forehead_bbox (todas en geometry.py).
def extract_regions_v2(image_rgb: np.ndarray, landmarks: np.ndarray):
    """
    Extrae 12 regiones faciales como bbox + crop RGB.

    Parámetros:
        image_rgb: imagen RGB (H, W, 3) ya alineada.
        landmarks: ndarray (N, 2) de landmarks densos en píxeles.

    Devuelve:
        dict ordenado region_name -> {bbox, crop_rgb, landmark_idx, source}.
    """
    # Mapa nombre_de_region -> índices de landmarks. La frente NO está
    # acá porque tiene su propio cálculo (no usa min/max sobre índices).
    region_defs = {
        "left_eye": LEFT_EYE_IDX,
        "right_eye": RIGHT_EYE_IDX,
        "mouth": LIPS_IDX,
        "nose": NOSE_IDX,
        "left_eyebrow": LEFT_EYEBROW_IDX,
        "right_eyebrow": RIGHT_EYEBROW_IDX,
        "left_cheekbone": LEFT_CHEEKBONE_IDX,
        "right_cheekbone": RIGHT_CHEEKBONE_IDX,
        "left_cheek": LEFT_CHEEK_IDX,
        "right_cheek": RIGHT_CHEEK_IDX,
        "chin": CHIN_IDX,
    }

    out = {}

    for region_name, idx_list in region_defs.items():
        # Mentón: bbox refinada — usa labios como referencia para el borde
        # superior, evita invadir la boca y el cuello.
        if region_name == "chin":
            bbox = get_chin_bbox_refined(
                landmarks=landmarks,
                image_shape=image_rgb.shape,
                chin_idx=CHIN_IDX,
                lips_idx=LIPS_IDX,
                side_pad=0.18,
                bottom_pad=0.10,
                top_offset_from_mouth=0.55,
            )
            crop = crop_from_bbox(image_rgb, bbox)
            out[region_name] = {
                "bbox": bbox,
                "crop_rgb": crop,
                "landmark_idx": idx_list,
                "source": "approx",
            }
            # `continue` salta el camino "bbox por min/max" más abajo.
            continue

        # Padding por tipo de región:
        # - ojos y boca: 0.25 (rasgos pequeños que merecen contexto extra).
        # - nariz: 0.22 (intermedia).
        # - resto: 0.20.
        if region_name in ["left_eye", "right_eye", "mouth"]:
            pad = 0.25
        elif region_name == "nose":
            pad = 0.22
        else:
            pad = 0.20

        # Bbox + crop estándar a partir de min/max de los landmarks.
        bbox = get_region_bbox(landmarks, idx_list, image_rgb.shape, pad=pad)
        crop = crop_from_bbox(image_rgb, bbox)

        out[region_name] = {
            "bbox": bbox,
            "crop_rgb": crop,
            "landmark_idx": idx_list,
            # Las regiones "oficiales" vienen de los conjuntos cerrados
            # de MediaPipe (ojos y boca). El resto son aproximaciones.
            "source": "official" if region_name in ["left_eye", "right_eye", "mouth"] else "approx",
        }

    # Frente: fuera del loop principal porque usa `get_forehead_bbox`,
    # que no toma una lista de índices arbitraria sino una combinación
    # ad-hoc de cejas y ojos. `landmark_idx=None` lo refleja.
    forehead_bbox = get_forehead_bbox(landmarks, image_rgb.shape)
    out["forehead"] = {
        "bbox": forehead_bbox,
        "crop_rgb": crop_from_bbox(image_rgb, forehead_bbox),
        "landmark_idx": None,
        "source": "approx",
    }

    return out


# =========================================================
# 2) Conveniencia: aplicar a un selected_pair
# =========================================================
# Aplica `extract_regions_v2` a las dos caras alineadas del par y guarda
# el resultado en `selected_pair["regions_v2"]` con la estructura:
#     {"A": {region_name: dict}, "B": {region_name: dict}}
# Mutación in-place + devuelve el dict por conveniencia.
#
# Depende de: extract_regions_v2 (definida arriba en este mismo módulo).
def add_regions_v2_to_pair(selected_pair: dict) -> dict:
    """
    Añade `regions_v2` al `selected_pair` (in-place + return).

    Parámetros:
        selected_pair: dict con 'aligned_a', 'aligned_b', 'landmarks_a',
            'landmarks_b' ya calculados.

    Devuelve:
        El mismo `selected_pair`, ahora con 'regions_v2' = {"A": ..., "B": ...}.
    """
    aligned_a = selected_pair["aligned_a"]
    aligned_b = selected_pair["aligned_b"]
    landmarks_a = selected_pair["landmarks_a"]
    landmarks_b = selected_pair["landmarks_b"]

    # Extracción independiente para cada cara — ningún acoplamiento entre A y B.
    regions_a = extract_regions_v2(aligned_a, landmarks_a)
    regions_b = extract_regions_v2(aligned_b, landmarks_b)

    selected_pair["regions_v2"] = {
        "A": regions_a,
        "B": regions_b,
    }

    return selected_pair
