# =========================================
# ID: PHYLOFACE_REGIONS_002
# VERSION: v1.0
# =========================================
# Origen de las funciones (migración Tarea #1):
#   src/phyloface_experimental_functions.py
#   - create_region_mask_from_points  (línea 1016)
#   - crop_mask_and_image             (línea 1047)
#   - extract_regions_v2_masked       (línea 1116)
#   - add_regions_v2_masked_to_pair   (línea 1287)
#
# Continuación del subsistema iniciado en `PHYLOFACE_REGIONS_001` (geometry +
# extract_rect). Esta entrega añade el camino **enmascarado**: además de la
# bbox y el crop rectangular, cada región trae una **máscara binaria** (uint8
# 0/255) que recorta la forma anatómica fiel (polígono o convex hull),
# y un `crop_masked_rgb` con el fondo a 0 fuera de la máscara.
#
# Estructura por región (contrato que viz/ y comparator_regional/ consumen):
#   {
#     "bbox":           (x1,y1,x2,y2)         # idéntica al path rect
#     "mask":           ndarray (H, W) uint8   # máscara en coords de la imagen completa
#     "crop_rgb":       ndarray (h, w, 3)      # recorte rectangular
#     "crop_mask":      ndarray (h, w)         # máscara recortada a la bbox
#     "crop_masked_rgb":ndarray (h, w, 3)      # crop con fondo a 0 fuera de máscara
#     "landmark_idx":   list[int] | None
#     "polygon_idx":    list[int] | None
#     "source":         "official" | "approx"
#   }
#
# Estrategia de máscara por región:
#   - Si la región tiene `polygon_idx` definido (ojos, boca, cejas, nariz):
#     se usan esos puntos como contorno cerrado y se rellena con `cv2.fillPoly`.
#   - Si no (pómulos, mejillas, mentón): se calcula el convex hull de los
#     landmarks de `landmark_idx` y se rellena ese polígono.
#   - Frente: caso especial, máscara = rectángulo de la bbox completa
#     (no hay contorno anatómico cerrado para la frente).
#
# Nota importante:
#   `extract_regions_v2_masked` **sobrescribe** la clave `regions_v2` del
#   `selected_pair` si existiera previamente. Esto es intencional y compatible
#   con el flujo del notebook original: primero se llama `add_regions_v2_to_pair`
#   (path rect) y después `add_regions_v2_masked_to_pair` (path masked), y este
#   último deja la versión enriquecida.

# -----------------------------------------
# FILE: phyloface/regions/extract_masked.py
# -----------------------------------------

import cv2
import numpy as np

from phyloface.regions.geometry import (
    # Subconjuntos de índices por región
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
    # Contornos poligonales ordenados
    LEFT_EYE_POLYGON_IDX,
    RIGHT_EYE_POLYGON_IDX,
    MOUTH_POLYGON_IDX,
    LEFT_EYEBROW_POLYGON_IDX,
    RIGHT_EYEBROW_POLYGON_IDX,
    NOSE_POLYGON_IDX,
    # Helpers geométricos
    get_region_bbox,
    get_forehead_bbox,
    get_chin_bbox_refined,
)


# =========================================================
# 1) Helper: máscara binaria a partir de polígono o convex hull
# =========================================================
# Genera una máscara 2D uint8 (0 fuera, 255 dentro) sobre todo el tamaño
# de la imagen original. La región puede definirse de dos formas:
#   a) `polygon_idx`: lista ordenada de índices que trazan el polígono
#      directamente — se usa si tiene 3 o más puntos.
#   b) `landmark_idx`: lista de índices del subconjunto de landmarks de
#      la región — se calcula el convex hull y se usa ese polígono.
# Si ninguna lista es válida (None o <3 puntos), devuelve una máscara vacía.
# No depende de funciones propias del módulo.
def create_region_mask_from_points(
    image_shape: tuple,
    landmarks: np.ndarray,
    polygon_idx: list[int] | None = None,
    landmark_idx: list[int] | None = None,
):
    """
    Genera una máscara binaria 2D para una región facial.

    Parámetros:
        image_shape: shape de la imagen completa (H, W[, C]).
        landmarks: ndarray (N, 2) de landmarks en píxeles.
        polygon_idx: índices ordenados que trazan el polígono (preferido).
        landmark_idx: índices del subconjunto de la región (fallback,
            se usa con convex hull).

    Devuelve:
        ndarray (H, W) uint8 con valores 0 o 255.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if polygon_idx is not None and len(polygon_idx) >= 3:
        # Camino preferido: usar el contorno ordenado tal como se definió.
        # int32 es el dtype que espera fillPoly.
        pts = landmarks[polygon_idx].astype(np.int32)
    elif landmark_idx is not None and len(landmark_idx) >= 3:
        # Fallback: convex hull sobre el subconjunto disponible. Convex hull
        # garantiza un polígono cerrado válido aunque los puntos no estén
        # ordenados anatómicamente.
        pts = landmarks[landmark_idx].astype(np.int32)
        pts = cv2.convexHull(pts)
    else:
        # Sin información suficiente: máscara vacía. El llamador decide
        # cómo manejarlo (típicamente el `crop_masked_rgb` queda en negro).
        return mask

    # fillPoly espera lista de contornos (de ahí los corchetes en [pts]).
    cv2.fillPoly(mask, [pts], 255)
    return mask


# =========================================================
# 2) Helper: recorte simultáneo de imagen, máscara y enmascarado
# =========================================================
# Recibe imagen + máscara (ambas en coordenadas globales) + bbox.
# Devuelve los tres recortes alineados a la bbox: el crop RGB rectangular,
# el crop de la máscara, y el crop RGB con fondo a 0 fuera de la máscara.
# Se usa para tener listos los tres formatos visuales/metrics típicos.
# No depende de funciones propias del módulo.
def crop_mask_and_image(image_rgb: np.ndarray, mask: np.ndarray, bbox: tuple):
    """
    Recorta simultáneamente la imagen, la máscara y el enmascarado.

    Parámetros:
        image_rgb: imagen RGB completa (H, W, 3).
        mask: máscara binaria completa (H, W).
        bbox: tupla (x1, y1, x2, y2).

    Devuelve:
        crop_rgb       : recorte rectangular (h, w, 3).
        crop_mask      : recorte de la máscara (h, w).
        crop_masked_rgb: recorte con fondo a 0 fuera de la máscara (h, w, 3).
    """
    x1, y1, x2, y2 = bbox

    # `.copy()` para que los recortes sean independientes del array original
    # (importante: el llamador puede modificar los crops sin tocar la imagen).
    crop_rgb = image_rgb[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2].copy()

    # Versión enmascarada: arrancamos del crop rectangular y zeroificamos
    # todo lo que esté fuera de la máscara (mask == 0). Esto deja la silueta
    # de la región sobre fondo negro, útil para visualización y para que
    # las métricas regionales no se contaminen con píxeles vecinos.
    crop_masked_rgb = crop_rgb.copy()
    crop_masked_rgb[crop_mask == 0] = 0

    return crop_rgb, crop_mask, crop_masked_rgb


# =========================================================
# 3) Extractor enmascarado (cara única)
# =========================================================
# Para cada región estándar:
#   - calcula la bbox (igual que el path rect, con el mismo padding).
#   - calcula la máscara (polygon_idx si existe; si no, convex hull).
#   - recorta los tres formatos con `crop_mask_and_image`.
# Mentón y frente tienen tratamiento especial (mentón usa bbox refinada;
# frente usa rectángulo como máscara).
#
# Depende de: create_region_mask_from_points, crop_mask_and_image,
#             get_region_bbox, get_chin_bbox_refined, get_forehead_bbox.
def extract_regions_v2_masked(image_rgb: np.ndarray, landmarks: np.ndarray):
    """
    Extrae 12 regiones faciales como bbox + máscara + crop rect + crop masked.

    Parámetros:
        image_rgb: imagen RGB (H, W, 3) ya alineada.
        landmarks: ndarray (N, 2) de landmarks densos en píxeles.

    Devuelve:
        dict region_name -> {bbox, mask, crop_rgb, crop_mask,
                             crop_masked_rgb, landmark_idx, polygon_idx, source}.
    """
    # Configuración por región. Cada entrada define:
    #   - landmark_idx : subconjunto para calcular la bbox.
    #   - polygon_idx  : contorno ordenado para la máscara (puede ser None;
    #                    en ese caso se usa convex hull de landmark_idx).
    #   - source       : "official" si la región viene de FaceMesh, "approx"
    #                    si es manual.
    #   - pad          : padding relativo para la bbox.
    region_defs = {
        "left_eye": {
            "landmark_idx": LEFT_EYE_IDX,
            "polygon_idx": LEFT_EYE_POLYGON_IDX,
            "source": "official",
            "pad": 0.25,
        },
        "right_eye": {
            "landmark_idx": RIGHT_EYE_IDX,
            "polygon_idx": RIGHT_EYE_POLYGON_IDX,
            "source": "official",
            "pad": 0.25,
        },
        "mouth": {
            "landmark_idx": LIPS_IDX,
            "polygon_idx": MOUTH_POLYGON_IDX,
            "source": "official",
            "pad": 0.25,
        },
        "nose": {
            "landmark_idx": NOSE_IDX,
            "polygon_idx": NOSE_POLYGON_IDX,
            "source": "approx",
            "pad": 0.22,
        },
        "left_eyebrow": {
            "landmark_idx": LEFT_EYEBROW_IDX,
            "polygon_idx": LEFT_EYEBROW_POLYGON_IDX,
            "source": "approx",
            "pad": 0.20,
        },
        "right_eyebrow": {
            "landmark_idx": RIGHT_EYEBROW_IDX,
            "polygon_idx": RIGHT_EYEBROW_POLYGON_IDX,
            "source": "approx",
            "pad": 0.20,
        },
        # Las siguientes 4 no tienen contorno ordenado definido →
        # cae al convex hull del landmark_idx.
        "left_cheekbone": {
            "landmark_idx": LEFT_CHEEKBONE_IDX,
            "polygon_idx": None,
            "source": "approx",
            "pad": 0.20,
        },
        "right_cheekbone": {
            "landmark_idx": RIGHT_CHEEKBONE_IDX,
            "polygon_idx": None,
            "source": "approx",
            "pad": 0.20,
        },
        "left_cheek": {
            "landmark_idx": LEFT_CHEEK_IDX,
            "polygon_idx": None,
            "source": "approx",
            "pad": 0.20,
        },
        "right_cheek": {
            "landmark_idx": RIGHT_CHEEK_IDX,
            "polygon_idx": None,
            "source": "approx",
            "pad": 0.20,
        },
    }

    out = {}

    # ---- Regiones estándar ----
    for region_name, cfg in region_defs.items():
        # Bbox: misma lógica que el path rect.
        bbox = get_region_bbox(
            landmarks=landmarks,
            idx_list=cfg["landmark_idx"],
            image_shape=image_rgb.shape,
            pad=cfg["pad"],
        )

        # Máscara: polygon_idx si lo hay; si no, convex hull de landmark_idx.
        mask = create_region_mask_from_points(
            image_shape=image_rgb.shape,
            landmarks=landmarks,
            polygon_idx=cfg["polygon_idx"],
            landmark_idx=cfg["landmark_idx"],
        )

        crop_rgb, crop_mask, crop_masked_rgb = crop_mask_and_image(
            image_rgb=image_rgb,
            mask=mask,
            bbox=bbox,
        )

        out[region_name] = {
            "bbox": bbox,
            "mask": mask,
            "crop_rgb": crop_rgb,
            "crop_mask": crop_mask,
            "crop_masked_rgb": crop_masked_rgb,
            "landmark_idx": cfg["landmark_idx"],
            "polygon_idx": cfg["polygon_idx"],
            "source": cfg["source"],
        }

    # ---- Mentón: bbox refinada + máscara por convex hull ----
    chin_bbox = get_chin_bbox_refined(
        landmarks=landmarks,
        image_shape=image_rgb.shape,
        chin_idx=CHIN_IDX,
        lips_idx=LIPS_IDX,
        side_pad=0.18,
        bottom_pad=0.10,
        top_offset_from_mouth=0.55,
    )
    chin_mask = create_region_mask_from_points(
        image_shape=image_rgb.shape,
        landmarks=landmarks,
        polygon_idx=None,
        landmark_idx=CHIN_IDX,
    )
    chin_crop_rgb, chin_crop_mask, chin_crop_masked_rgb = crop_mask_and_image(
        image_rgb=image_rgb,
        mask=chin_mask,
        bbox=chin_bbox,
    )
    out["chin"] = {
        "bbox": chin_bbox,
        "mask": chin_mask,
        "crop_rgb": chin_crop_rgb,
        "crop_mask": chin_crop_mask,
        "crop_masked_rgb": chin_crop_masked_rgb,
        "landmark_idx": CHIN_IDX,
        "polygon_idx": None,
        "source": "approx",
    }

    # ---- Frente: bbox aproximada + máscara = rectángulo completo ----
    # No hay contorno anatómico cerrado para la frente, así que la máscara
    # cubre toda la bbox. Eso hace que `crop_masked_rgb == crop_rgb` para
    # esta región, pero mantiene la estructura del dict uniforme.
    forehead_bbox = get_forehead_bbox(landmarks, image_rgb.shape)
    forehead_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = forehead_bbox
    forehead_mask[y1:y2, x1:x2] = 255

    forehead_crop_rgb, forehead_crop_mask, forehead_crop_masked_rgb = crop_mask_and_image(
        image_rgb=image_rgb,
        mask=forehead_mask,
        bbox=forehead_bbox,
    )
    out["forehead"] = {
        "bbox": forehead_bbox,
        "mask": forehead_mask,
        "crop_rgb": forehead_crop_rgb,
        "crop_mask": forehead_crop_mask,
        "crop_masked_rgb": forehead_crop_masked_rgb,
        "landmark_idx": None,
        "polygon_idx": None,
        "source": "approx",
    }

    return out


# =========================================================
# 4) Conveniencia: aplicar a un selected_pair
# =========================================================
# Reemplaza `selected_pair["regions_v2"]` con la versión enriquecida (con
# máscara) para A y B. La estructura externa es la misma del path rect
# ({"A": ..., "B": ...}), pero el dict de cada región trae las claves
# extra: `mask`, `crop_mask`, `crop_masked_rgb`, `polygon_idx`.
#
# Depende de: extract_regions_v2_masked (definida arriba en este módulo).
def add_regions_v2_masked_to_pair(selected_pair: dict) -> dict:
    """
    Añade `regions_v2` enmascarado al `selected_pair` (in-place + return).

    Si `regions_v2` ya estaba (del path rect), se sobrescribe con la
    versión enriquecida con máscara.

    Parámetros:
        selected_pair: dict con 'aligned_a', 'aligned_b', 'landmarks_a',
            'landmarks_b' ya calculados.

    Devuelve:
        El mismo `selected_pair`, con 'regions_v2' = {"A": ..., "B": ...}.
    """
    aligned_a = selected_pair["aligned_a"]
    aligned_b = selected_pair["aligned_b"]
    landmarks_a = selected_pair["landmarks_a"]
    landmarks_b = selected_pair["landmarks_b"]

    regions_a = extract_regions_v2_masked(aligned_a, landmarks_a)
    regions_b = extract_regions_v2_masked(aligned_b, landmarks_b)

    selected_pair["regions_v2"] = {
        "A": regions_a,
        "B": regions_b,
    }

    return selected_pair
