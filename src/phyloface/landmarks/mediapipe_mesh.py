# =========================================
# ID: PHYLOFACE_LANDMARKS_001
# VERSION: v1.0
# =========================================
# Origen de las funciones (migración Tarea #1):
#   src/phyloface_experimental_functions.py
#   - init_face_mesh             (línea 505)
#   - get_face_mesh_landmarks    (línea 526)
#   - add_dense_landmarks_to_pair (línea 550)
#
# Qué hace este módulo:
# - Encapsula el backend MediaPipe Face Mesh como fuente de landmarks
#   densos del rostro (468 puntos 3D; aquí trabajamos solo con las coords
#   X, Y en píxeles del recorte alineado).
# - Provee 3 funciones de uso típico:
#     1) `init_face_mesh`            -> construye el objeto FaceMesh listo para usar.
#     2) `get_face_mesh_landmarks`   -> corre la inferencia sobre una imagen RGB
#                                      y devuelve los puntos en coordenadas de píxel.
#     3) `add_dense_landmarks_to_pair` -> aplica (2) a las dos caras alineadas de
#                                      un `selected_pair` y guarda los resultados
#                                      como `landmarks_a` / `landmarks_b`.
#
# Decisiones de diseño:
# - Se asume `static_image_mode=True` por defecto: este pipeline procesa
#   fotos sueltas (no video). Para video conviene desactivarlo y dejar que
#   MediaPipe mantenga tracking entre frames.
# - `refine_landmarks=True` agrega puntos extra de iris y labios (los famosos
#   478 totales en vez de 468). Acá se deja activado porque mejora la
#   precisión de las regiones perioculares y de la boca.
# - Si MediaPipe no detecta cara, se lanza ValueError explícito. El usuario
#   debería revisar la calidad/alineación del crop antes de seguir.
#
# Cosas que NO hace este módulo (a propósito):
# - No visualiza los landmarks. La función `plot_face_with_landmarks` que
#   acompañaba a estas en el archivo original vive ahora en `viz/landmarks.py`.
# - No clasifica los landmarks en regiones. Eso es responsabilidad del
#   subpaquete `regions/`, que consume estos arrays.

# -----------------------------------------
# FILE: phyloface/landmarks/mediapipe_mesh.py
# -----------------------------------------

import numpy as np
import mediapipe as mp


# =========================================================
# 1) Inicialización del modelo MediaPipe Face Mesh
# =========================================================
# Construye el objeto FaceMesh con parámetros adecuados para el caso típico
# del proyecto: imágenes estáticas, una sola cara por crop alineado.
# Es el punto de entrada antes de cualquier otra función del módulo.
# No depende de funciones propias del módulo.
def init_face_mesh(
    static_image_mode: bool = True,
    max_num_faces: int = 1,
    refine_landmarks: bool = True,
    min_detection_confidence: float = 0.5,
):
    """
    Inicializa MediaPipe Face Mesh y devuelve el objeto listo para inferir.

    Parámetros:
        static_image_mode: True para fotos sueltas; False para tracking en video.
        max_num_faces: cuántas caras detectar como máximo (1 = una sola).
        refine_landmarks: True añade puntos extra de iris y labios (478 en total).
        min_detection_confidence: umbral mínimo (0..1) para aceptar la detección.

    Devuelve:
        Una instancia de `mp.solutions.face_mesh.FaceMesh`.
    """
    # `mp.solutions.face_mesh` es el módulo de MediaPipe que expone el constructor.
    # Se referencia local para que la dependencia quede explícita y aislada.
    mp_face_mesh = mp.solutions.face_mesh

    return mp_face_mesh.FaceMesh(
        static_image_mode=static_image_mode,
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
    )


# =========================================================
# 2) Inferencia: imagen RGB -> array de landmarks en píxeles
# =========================================================
# Toma una imagen ya alineada (RGB, H x W x 3) y devuelve la matriz Nx2
# con las coordenadas (x, y) en píxeles de los N landmarks densos.
# Si MediaPipe no detecta una cara en la imagen, lanza ValueError —
# eso suele indicar que el crop facial no es lo suficientemente claro
# o que la alineación previa salió mal.
# No depende de funciones propias del módulo.
def get_face_mesh_landmarks(face_mesh, image_rgb: np.ndarray) -> np.ndarray:
    """
    Ejecuta Face Mesh sobre una imagen y devuelve landmarks en píxeles.

    Parámetros:
        face_mesh: objeto devuelto por `init_face_mesh`.
        image_rgb: imagen RGB (H, W, 3) como ndarray uint8.

    Devuelve:
        ndarray float32 de forma (N, 2) con coordenadas (x, y) en píxeles.
        N suele ser 468 o 478 según `refine_landmarks`.
    """
    # Tamaño en píxeles del crop. MediaPipe devuelve coordenadas normalizadas
    # en [0, 1], así que multiplicamos por (w, h) para volver a píxeles.
    h, w = image_rgb.shape[:2]

    # MediaPipe espera la imagen en RGB (no BGR). Acá ya recibimos RGB,
    # cosa que viene garantizada río arriba por `load_image`.
    result = face_mesh.process(image_rgb)

    # `multi_face_landmarks` es None o lista vacía si no detectó nada.
    # No tiene sentido seguir sin landmarks; cortamos con un error claro.
    if not result.multi_face_landmarks:
        raise ValueError("No se detectaron landmarks faciales con MediaPipe.")

    # Como `max_num_faces=1` en init_face_mesh por defecto, nos quedamos con
    # el primer (y único) conjunto. Si en el futuro se permite multi-cara,
    # esta línea hay que repensarla.
    face_landmarks = result.multi_face_landmarks[0]

    # Convertimos a lista de pares (x, y) en píxeles enteros (float32 para
    # poder hacer cuentas geométricas con ellos más adelante sin perder
    # precisión sub-pixel).
    points = []
    for lm in face_landmarks.landmark:
        x = lm.x * w
        y = lm.y * h
        points.append([x, y])

    return np.array(points, dtype=np.float32)


# =========================================================
# 3) Conveniencia: añadir landmarks a un selected_pair
# =========================================================
# Aplica `get_face_mesh_landmarks` a las dos caras alineadas de un par
# (`aligned_a` / `aligned_b`) y guarda los arrays resultantes dentro del
# mismo dict bajo `landmarks_a` / `landmarks_b`. Devuelve el dict mutado
# por conveniencia, aunque la mutación es in-place.
# Depende de: `get_face_mesh_landmarks` (definida arriba en este mismo módulo).
def add_dense_landmarks_to_pair(face_mesh, selected_pair: dict) -> dict:
    """
    Añade landmarks densos a un selected_pair (in-place + return).

    Parámetros:
        face_mesh: objeto devuelto por `init_face_mesh`.
        selected_pair: dict con al menos las claves 'aligned_a' y 'aligned_b'.

    Devuelve:
        El mismo `selected_pair`, ahora con 'landmarks_a' y 'landmarks_b'.
    """
    # Sacamos los dos crops faciales ya alineados. La alineación previa
    # asegura que los landmarks queden en posiciones comparables entre A y B.
    aligned_a = selected_pair["aligned_a"]
    aligned_b = selected_pair["aligned_b"]

    # Inferimos los landmarks de cada cara. Si alguno falla, el ValueError
    # de `get_face_mesh_landmarks` se propaga tal cual (no lo silenciamos).
    selected_pair["landmarks_a"] = get_face_mesh_landmarks(face_mesh, aligned_a)
    selected_pair["landmarks_b"] = get_face_mesh_landmarks(face_mesh, aligned_b)

    return selected_pair
