# =========================================
# ID: PHYLOFACE_CORE_BASE_001
# VERSION: v1.0
# =========================================
# Origen de la función (migración Tarea #1, Paso 6a):
#   src/phyloface_experimental_functions.py
#   - load_image  (línea 42)
#
# Esta entrega cubre 2 archivos (mismo ID):
#   1) phyloface/core/io.py        (este archivo, nuevo)
#   2) phyloface/core/metrics.py   (reescrito en esta entrega: ver allí)
#
# Qué hace este módulo:
# - Provee `load_image`: utilidad para abrir una imagen del disco y
#   devolverla en RGB (OpenCV abre por defecto en BGR).
# - Hace validaciones explícitas y emite mensajes de error útiles
#   para diagnosticar problemas de path (file not found, cwd, etc.)
#   y de lectura (archivo corrupto o formato no soportado por OpenCV).
#
# Decisiones de diseño:
# - Devuelve **ndarray uint8 RGB** (H, W, 3). Es lo que esperan los
#   módulos río abajo (`phyloface.regions`, `phyloface.landmarks`,
#   `phyloface.core.detector`).
# - No hace resize ni normalización: es un loader "crudo". Cualquier
#   transformación posterior (alineación, padding, normalización) es
#   responsabilidad del consumidor.
#
# Cosas que NO hace este módulo:
# - No procesa lotes (use list comprehensions del lado del llamador).
# - No detecta caras, no extrae embeddings — eso vive en `pairs.py` y
#   `embedder.py` (siguientes sub-pasos 6b y 6c).

# -----------------------------------------
# FILE: phyloface/core/io.py
# -----------------------------------------

from pathlib import Path

import cv2
import numpy as np


# =========================================================
# 1) Loader de imagen: disco -> ndarray RGB
# =========================================================
# Recibe un str o Path y devuelve la imagen como ndarray (H, W, 3) en RGB.
# - Si el archivo no existe, lanza FileNotFoundError con detalle de las
#   tres rutas relevantes (la dada, la resuelta y el cwd) — esto es la
#   trampa más común de paths relativos al correr desde notebook vs CLI.
# - Si el archivo existe pero OpenCV no puede leerlo (formato no soportado,
#   archivo corrupto, permisos), lanza ValueError con la ruta resuelta.
# - Si todo va bien, convierte BGR (default de OpenCV) a RGB y devuelve.
# No depende de funciones propias del módulo.
def load_image(image_path: str | Path) -> np.ndarray:
    """
    Carga una imagen desde disco y la convierte a RGB.

    Parámetros:
        image_path: ruta como str o Path.

    Devuelve:
        ndarray uint8 con forma (H, W, 3) en orden RGB.

    Lanza:
        FileNotFoundError: si el archivo no existe.
        ValueError: si OpenCV no puede leer el archivo.
    """
    image_path = Path(image_path)
    image_path_resolved = image_path.resolve()

    if not image_path.exists():
        raise FileNotFoundError(
            f"No existe el archivo:\n"
            f"  ruta dada: {image_path}\n"
            f"  ruta resuelta: {image_path_resolved}\n"
            f"  cwd actual: {Path.cwd()}"
        )

    # OpenCV lee en BGR por defecto.
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(
            f"El archivo existe pero OpenCV no pudo leerlo:\n"
            f"  ruta: {image_path_resolved}"
        )

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
