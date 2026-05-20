# =========================================
# ID: PHYLOFACE_REGIONS_001
# VERSION: v1.0
# =========================================
# Origen de las funciones y constantes (migración Tarea #1):
#   src/phyloface_experimental_functions.py
#   - connection_set_to_index_list  (línea 588)
#   - constantes LEFT_EYE_IDX ... CHIN_IDX   (líneas 599-617)
#   - get_region_bbox               (línea 623)
#   - crop_from_bbox                (línea 651)
#   - get_forehead_bbox             (línea 662)
#   - get_chin_bbox_refined         (línea 694)
#
# Qué hace este módulo:
# - Define la **lista canónica de índices de landmarks por región** que se
#   usa en todo el pipeline regional. Las constantes vienen de dos fuentes:
#     a) regiones "oficiales" de MediaPipe (ojos y labios) — se derivan
#        a partir de los conjuntos de conexiones (`FACEMESH_LEFT_EYE`, etc.)
#        usando `connection_set_to_index_list`.
#     b) regiones "aproximadas" (cejas, nariz, pómulos, mejillas, mentón) —
#        listas manuales de índices que la charla externa con ChatGPT
#        recomendó como subconjuntos razonables para empezar.
# - Provee 4 helpers de geometría que operan sobre arrays de landmarks:
#     get_region_bbox   : bbox por min/max + padding.
#     crop_from_bbox    : recorta la imagen por una bbox dada.
#     get_forehead_bbox : aproxima la frente (no hay región oficial cerrada).
#     get_chin_bbox_refined : recorte de mentón con ajustes anatómicos.
#
# Decisiones de diseño:
# - Las constantes se evalúan **al importar el módulo** (no son lazy).
#   Esto significa que `import mediapipe` se ejecuta de entrada. Es
#   intencional: si MediaPipe no está disponible el error aparece pronto.
# - Las regiones aproximadas se dejan tal cual el archivo original.
#   Re-validar/refinar los subconjuntos es trabajo de Tarea #2 del proyecto
#   ("formalizar lista canónica de regiones") — esta migración no los toca
#   para no introducir cambios funcionales encubiertos.
#
# Cosas que NO hace este módulo:
# - No extrae regiones (eso está en `extract_rect.py` y `extract_masked.py`).
# - No visualiza (eso está en `viz/regions.py`).

# -----------------------------------------
# FILE: phyloface/regions/geometry.py
# -----------------------------------------

import numpy as np
import mediapipe as mp


# Acceso al módulo Face Mesh de MediaPipe. Se referencia local para que la
# dependencia quede explícita en este archivo y no se mezcle con otras.
mp_face_mesh = mp.solutions.face_mesh


# =========================================================
# 1) Helper: conjunto de conexiones -> lista ordenada de índices
# =========================================================
# MediaPipe expone las regiones oficiales (ojos, labios, etc.) como
# **conjuntos de aristas** entre landmarks: pares (a, b) que definen un
# polígono cerrado. Para nuestros usos (subconjunto de puntos) alcanza con
# la lista plana de índices únicos, ordenada.
# No depende de funciones propias del módulo.
def connection_set_to_index_list(connection_set):
    """
    Convierte un conjunto de conexiones (pares de índices) en una lista
    ordenada de índices únicos.

    Parámetros:
        connection_set: iterable de tuplas (a, b), formato MediaPipe.

    Devuelve:
        list[int] ordenada de forma ascendente, sin duplicados.
    """
    idx = set()
    for a, b in connection_set:
        idx.add(a)
        idx.add(b)
    return sorted(idx)


# =========================================================
# 2) Constantes: índices de landmarks por región
# =========================================================
# Regiones "oficiales" (derivadas de los conjuntos oficiales de MediaPipe).
# Se evalúan en tiempo de import: si en algún día actualizan los índices
# en MediaPipe, estos constantes los reflejan automáticamente.
LEFT_EYE_IDX = connection_set_to_index_list(mp_face_mesh.FACEMESH_LEFT_EYE)
RIGHT_EYE_IDX = connection_set_to_index_list(mp_face_mesh.FACEMESH_RIGHT_EYE)
LIPS_IDX = connection_set_to_index_list(mp_face_mesh.FACEMESH_LIPS)

# Regiones "aproximadas" — listas manuales heredadas del archivo original.
# Cada lista es un subconjunto razonable de landmarks anatómicos para esa
# región, suficiente para calcular bboxes coherentes; no son exhaustivas.

# Nariz: puente + alas + columela. Mezcla landmarks centrales y laterales.
NOSE_IDX = sorted(set([
    1, 2, 4, 5, 6, 19, 20, 45, 48, 49, 64, 94, 97, 98,
    115, 122, 129, 168, 195, 197, 218, 275, 278, 279, 294, 327, 331, 344
]))

# Cejas: arcos superciliares izquierdo y derecho.
LEFT_EYEBROW_IDX = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW_IDX = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]

# Pómulos: zona ósea bajo el ojo (más arriba/lateral que la mejilla).
LEFT_CHEEKBONE_IDX = [50, 101, 100, 126, 142, 203, 206]
RIGHT_CHEEKBONE_IDX = [280, 330, 329, 355, 371, 423, 426]

# Mejillas: superficie media de la cara, por debajo del pómulo.
LEFT_CHEEK_IDX = [116, 117, 118, 119, 120, 100, 126, 142, 203, 205, 50]
RIGHT_CHEEK_IDX = [345, 346, 347, 348, 349, 329, 355, 371, 423, 425, 280]

# Mentón: zona inferior de la cara, desde la línea mandibular hasta la
# punta. Más adelante se refina con `get_chin_bbox_refined`, no con la
# bbox cruda de estos puntos.
CHIN_IDX = [152, 148, 176, 149, 150, 136, 172, 58, 132, 361, 397, 378, 379, 365, 288, 435]


# =========================================================
# 2b) Constantes: contornos poligonales por región
# =========================================================
# Las listas `*_IDX` de arriba son **subconjuntos** de landmarks útiles
# para calcular bboxes y máscaras por convex hull. Para las regiones
# donde queremos una máscara poligonal *anatómicamente más fiel*
# (ojos, boca, cejas, nariz), definimos también un **contorno ordenado**:
# una secuencia de índices que, recorridos en orden, trazan el polígono.
#
# Origen: estas listas vienen del bloque `PHYLOFACE_REGIONS_MASK_001`
# del archivo experimental original (líneas 1000-1010). Son las que usa
# `extract_regions_v2_masked` para llenar máscaras con `cv2.fillPoly`.
#
# Regiones sin `*_POLYGON_IDX` definido (pómulos, mejillas, mentón,
# frente) se enmascaran por otra vía:
#   - pómulos / mejillas / mentón: convex hull del `*_IDX` correspondiente.
#   - frente: rectángulo cubriendo toda la bbox (no hay contorno cerrado).

# Ojos: contorno ordenado del párpado externo (6 puntos, suficiente
# para un polígono cerrado razonable).
LEFT_EYE_POLYGON_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_POLYGON_IDX = [362, 385, 387, 263, 373, 380]

# Boca: contorno externo de los labios (11 puntos).
MOUTH_POLYGON_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

# Cejas: arcos superciliares (mismas listas que `*_EYEBROW_IDX`,
# duplicadas a propósito para dejar explícito que acá se interpretan
# como **secuencia ordenada** para fillPoly, no como subconjunto).
LEFT_EYEBROW_POLYGON_IDX = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW_POLYGON_IDX = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]

# Nariz: contorno externo aproximado (puente + alas + columela).
NOSE_POLYGON_IDX = [168, 6, 197, 195, 5, 4, 1, 94, 97, 2, 326, 327]


# =========================================================
# 3) Helper geométrico: bbox por min/max con padding
# =========================================================
# Caso general: dada una lista de índices, devuelve la bbox rectangular
# mínima que contiene esos puntos, expandida un porcentaje configurable
# en ambos ejes. Recorta a los límites de la imagen.
# No depende de funciones propias del módulo.
def get_region_bbox(
    landmarks: np.ndarray,
    idx_list: list[int],
    image_shape: tuple,
    pad: float = 0.20,
):
    """
    Bbox alrededor de un subconjunto de landmarks, con padding relativo.

    Parámetros:
        landmarks: ndarray (N, 2) de landmarks en píxeles.
        idx_list: lista de índices a usar para definir la región.
        image_shape: shape de la imagen (H, W[, C]) para clipping.
        pad: padding relativo al tamaño de la bbox (0.20 = 20%).

    Devuelve:
        Tupla (x1, y1, x2, y2) en píxeles, dentro de los límites de la imagen.
    """
    h, w = image_shape[:2]
    pts = landmarks[idx_list]

    # min/max de los puntos en X e Y.
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    # Ancho/alto de la bbox cruda. max(1.0, ...) evita división por 0
    # en regiones degeneradas (todos los puntos colapsados a uno solo).
    bw = max(1.0, x_max - x_min)
    bh = max(1.0, y_max - y_min)

    # Expansión proporcional al tamaño de la región: regiones grandes
    # crecen más, regiones chicas crecen menos. Recortado a [0, w/h].
    x1 = max(0, int(round(x_min - bw * pad)))
    y1 = max(0, int(round(y_min - bh * pad)))
    x2 = min(w, int(round(x_max + bw * pad)))
    y2 = min(h, int(round(y_max + bh * pad)))

    return (x1, y1, x2, y2)


# =========================================================
# 4) Helper: recorte rectangular a partir de una bbox
# =========================================================
# Utilidad simple, usada en todos los flujos de extracción. Devuelve una
# copia (no una vista) para que el recorte sea independiente del array
# original y pueda mutarse sin efectos colaterales.
# No depende de funciones propias del módulo.
def crop_from_bbox(image_rgb: np.ndarray, bbox: tuple):
    """
    Recorta `image_rgb` por la bbox (x1, y1, x2, y2) y devuelve una copia.
    """
    x1, y1, x2, y2 = bbox
    return image_rgb[y1:y2, x1:x2].copy()


# =========================================================
# 5) Frente: aproximación a partir de cejas + ojos
# =========================================================
# La frente no está delimitada como región cerrada en Face Mesh. Acá se
# aproxima usando como referencia las cejas (límite inferior) y la
# distancia vertical cejas-ojos (estimador del alto de la frente).
# No depende de funciones propias del módulo.
def get_forehead_bbox(landmarks: np.ndarray, image_shape: tuple):
    """
    Bbox aproximada de la frente.

    Parámetros:
        landmarks: ndarray (N, 2) en píxeles.
        image_shape: shape de la imagen (H, W[, C]) para clipping.

    Devuelve:
        Tupla (x1, y1, x2, y2) en píxeles.
    """
    h, w = image_shape[:2]

    # Tomamos cejas y ojos como referencias anatómicas.
    brow_idx = LEFT_EYEBROW_IDX + RIGHT_EYEBROW_IDX
    eye_idx = LEFT_EYE_IDX + RIGHT_EYE_IDX

    brow_pts = landmarks[brow_idx]
    eye_pts = landmarks[eye_idx]

    # Límites horizontales: cubren todo el ancho de ambas cejas.
    x_min, y_min = brow_pts.min(axis=0)
    x_max, _ = brow_pts.max(axis=0)

    # Centros verticales (promedios en Y) de cejas y ojos. La diferencia
    # entre ellos da una escala vertical de la cara que se usa para
    # estimar cuánto subir desde las cejas para cubrir la frente.
    brow_center_y = brow_pts[:, 1].mean()
    eye_center_y = eye_pts[:, 1].mean()

    # Altura de la frente: el doble de la distancia centro-cejas a centro-ojos.
    # Se acota a un mínimo de 8 px para no degenerar en regiones casi vacías
    # cuando la cara está muy de perfil o muy chica.
    delta = max(8, int(round(abs(eye_center_y - brow_center_y) * 2.0)))

    # Bbox final: pequeño padding horizontal (8 px); verticalmente desde
    # justo arriba de las cejas hasta `delta` píxeles más arriba.
    x1 = max(0, int(round(x_min - 8)))
    x2 = min(w, int(round(x_max + 8)))
    y2 = max(0, int(round(y_min + 2)))
    y1 = max(0, y2 - delta)

    return (x1, y1, x2, y2)


# =========================================================
# 6) Mentón: bbox refinada (anatómicamente más razonable)
# =========================================================
# El recorte por min/max sobre `CHIN_IDX` invade boca y cuello. Esta
# función define un mentón "anatómicamente útil":
#   - límite superior: a mitad de camino entre el labio inferior y el
#     borde inferior del mentón (configurable con top_offset_from_mouth).
#   - límite inferior: borde inferior de los landmarks de mentón + padding.
#   - laterales: extremos de mentón + padding lateral.
# No depende de funciones propias del módulo.
def get_chin_bbox_refined(
    landmarks: np.ndarray,
    image_shape: tuple,
    chin_idx: list[int],
    lips_idx: list[int],
    side_pad: float = 0.18,
    bottom_pad: float = 0.10,
    top_offset_from_mouth: float = 0.55,
):
    """
    Bbox del mentón con ajustes anatómicos (evita invadir boca o cuello).

    Parámetros:
        landmarks: ndarray (N, 2) en píxeles.
        image_shape: shape de la imagen (H, W[, C]) para clipping.
        chin_idx: índices de landmarks del mentón.
        lips_idx: índices de landmarks de los labios.
        side_pad: padding lateral relativo al ancho del mentón.
        bottom_pad: padding inferior relativo al alto del mentón.
        top_offset_from_mouth: 0..1, cuánto bajar desde la boca al mentón
            para fijar el borde superior (0.55 ≈ un poco menos de la mitad
            por debajo del labio inferior).

    Devuelve:
        Tupla (x1, y1, x2, y2) en píxeles.
    """
    h, w = image_shape[:2]

    # Puntos del mentón y los labios.
    chin_pts = landmarks[chin_idx]
    lips_pts = landmarks[lips_idx]

    # min/max del mentón en ambos ejes. Se desempacan en orden:
    # x_min, y_min, x_max, y_max para tener todo a mano.
    chin_x_min, _, chin_x_max, chin_y_max = (
        chin_pts[:, 0].min(),
        chin_pts[:, 1].min(),
        chin_pts[:, 0].max(),
        chin_pts[:, 1].max(),
    )

    # Rango vertical de los labios: lo usamos como referencia para decidir
    # dónde empieza el mentón "útil".
    lips_y_min = lips_pts[:, 1].min()
    lips_y_max = lips_pts[:, 1].max()

    # Tamaños de referencia. max(1.0, ...) evita degeneración por
    # cara muy chica o landmarks colapsados.
    chin_w = max(1.0, chin_x_max - chin_x_min)
    lips_h = max(1.0, lips_y_max - lips_y_min)
    chin_h = max(1.0, chin_y_max - chin_pts[:, 1].min())

    # Borde superior: empieza un poco por debajo del labio inferior.
    # Con top_offset_from_mouth=0.55 el corte cae a 0.45 * alto_de_labios
    # por debajo del labio inferior.
    y1 = lips_y_max - (lips_h * (1.0 - top_offset_from_mouth))

    # Borde inferior: extremo del mentón + un poco de padding.
    y2 = chin_y_max + chin_h * bottom_pad

    # Laterales: extremos del mentón + padding proporcional.
    x1 = chin_x_min - chin_w * side_pad
    x2 = chin_x_max + chin_w * side_pad

    # Clipping final a límites de la imagen + conversión a int.
    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(w, int(round(x2)))
    y2 = min(h, int(round(y2)))

    return (x1, y1, x2, y2)
