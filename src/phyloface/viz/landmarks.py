# =========================================
# ID: PHYLOFACE_VIZ_001
# VERSION: v1.0
# =========================================
# Origen de la función (migración Tarea #1, Paso 7):
#   src/phyloface_experimental_functions.py
#   - plot_face_with_landmarks  (línea 566)
#
# Qué hace este módulo:
# - Visualización del bloque de **landmarks densos** (módulo compañero:
#   `phyloface.landmarks`). Una sola función `plot_face_with_landmarks`
#   que superpone una nube de puntos (los 468/478 landmarks de
#   MediaPipe Face Mesh) sobre una imagen RGB.
# - Sirve como control visual rápido para verificar que MediaPipe
#   ubicó correctamente los puntos en ojos, cejas, nariz, boca y
#   contorno antes de usarlos para definir regiones.
#
# Decisiones de diseño:
# - `s=3` (tamaño del marker en scatter) es el default del archivo
#   original — suficientemente chico para que se vea la silueta de los
#   468 puntos sin tapar la cara.
# - Sin color explícito: usa el ciclo de colores de matplotlib (rojo
#   por default en backend Agg).
# - No devuelve `fig`; misma decisión que en `detection.py` (ver allí).

# -----------------------------------------
# FILE: phyloface/viz/landmarks.py
# -----------------------------------------

import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1) Imagen + scatter de landmarks
# =========================================================
# Imagen RGB de fondo + scatter plot de los landmarks en coordenadas
# de píxel. Title configurable por el llamador (típicamente algo como
# "aligned_a + dense landmarks").
# No depende de funciones propias del módulo.
def plot_face_with_landmarks(image_rgb: np.ndarray, landmarks: np.ndarray, title: str):
    """
    Superpone landmarks sobre una imagen RGB.

    Parámetros:
        image_rgb: imagen RGB (H, W, 3) — típicamente un rostro alineado.
        landmarks: ndarray (N, 2) con coordenadas (x, y) en píxeles.
        title: título del plot.
    """
    plt.figure(figsize=(4, 4))
    plt.imshow(image_rgb)
    # `s=3` mantiene los markers chicos para que la silueta de los
    # 468/478 landmarks sea visible sin ocultar la cara.
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=3)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
