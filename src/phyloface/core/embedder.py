# =========================================
# ID: PHYLOFACE_EMBEDDER_001
# VERSION: v1.0
# =========================================
# Origen de las funciones (migración Tarea #1, Paso 6b):
#   src/phyloface_experimental_functions.py
#   - get_recognition_model          (línea 379)
#   - extract_embedding_from_aligned (línea 440)
#
# Qué hace este módulo:
# - Aísla el camino "rostro alineado → embedding facial" usando el
#   **submodelo de reconocimiento** de InsightFace. La diferencia clave con
#   `face.embedding` (que ya viene en el resultado de la detección) es que
#   acá se **re-extrae** el embedding sobre una cara que ya pasó por
#   alineación canónica nuestra (`align_face_from_record` en `pairs.py`),
#   con margen y tamaño controlados por el llamador.
#
# - Por qué importa: la detección de InsightFace usa su propia alineación
#   (5-puntos) sobre el crop original. Si después uno *re-alinea* la cara
#   (más margen, otro tamaño, ángulo controlado), el embedding original
#   ya no representa lo que ves en pantalla. Recalcularlo sobre la
#   versión alineada da consistencia.
#
# Decisiones de diseño:
# - `get_recognition_model` descubre el submodelo por **introspección**
#   (busca el primero con método `get_feat`). Funciona con varios modelos
#   de la familia InsightFace (`buffalo_l`, `buffalo_s`, etc.) sin tener
#   que hardcodear el nombre del submodelo.
# - Si el `FaceAnalysis` no tiene ningún modelo con `get_feat`, lanza
#   `RuntimeError` con mensaje explícito — no devuelve None silenciosamente.
# - `extract_embedding_from_aligned` siempre devuelve float32 aplanado.
#   `get_feat` puede devolver shapes 2D (1, 512); el flatten asegura
#   compatibilidad con `phyloface.core.metrics.l2_normalize` (que usa ravel).
#
# Cosas que NO hace este módulo:
# - No alinea caras (eso es `pairs.align_face_from_record`, Paso 6c).
# - No normaliza el embedding (eso lo hace `core.metrics.l2_normalize`
#   en el momento de comparar; el embedding crudo se guarda tal cual).
# - No inicializa el `FaceAnalysis` (eso es `pairs.init_face_app`, 6c).
#
# === BUGFIX 2026-05-20 (Spike Track 2a) ===
# Descubrimiento del Spike de paridad JS/Python: `get_feat` de InsightFace
# usa internamente `cv2.dnn.blobFromImages(..., swapRB=True)`, lo que
# significa que **asume input en BGR** (default de OpenCV) y convierte a
# RGB antes de pasar al modelo. El archivo experimental original (y por
# herencia esta función) le pasaba `aligned_rgb` (ya en RGB) → el swap
# interno lo invertía → el modelo recibía BGR → canales invertidos vs
# entrenamiento.
#
# El bug era **silencioso cualitativamente**: como ambas caras de un par
# sufrían la misma inversión, la similitud coseno entre ellas seguía
# midiendo "qué tan parecidas son las imágenes" de forma consistente; el
# ranking se preservaba. Pero el valor absoluto del embedding no era el
# "verdadero" del modelo, lo que invalida calibraciones contra umbrales
# externos o comparaciones contra embeddings de otros sistemas.
#
# Fix: convertir RGB→BGR antes de llamar `get_feat`, así el swap interno
# lo devuelve a RGB y el modelo recibe lo que esperaba ver en training.

import cv2
import numpy as np


# =========================================================
# 1) Localizar el submodelo de reconocimiento dentro del FaceAnalysis
# =========================================================
# InsightFace `FaceAnalysis` carga varios submodelos en el dict `.models`
# (detection, landmark_2d_106, landmark_3d_68, genderage, recognition).
# Sólo el de reconocimiento expone `get_feat(aligned_rgb) -> embedding`.
# Esta función itera buscando ese atributo para no tener que conocer la
# clave exacta del submodelo (varía entre modelos: 'recognition' en algunos,
# otro nombre en otros).
# No depende de funciones propias del módulo.
def get_recognition_model(face_app):
    """
    Devuelve el submodelo de reconocimiento de un FaceAnalysis.

    Parámetros:
        face_app: instancia de `insightface.app.FaceAnalysis` ya `prepare()`-ada.

    Devuelve:
        El submodelo que expone `get_feat(aligned_rgb)`.

    Lanza:
        RuntimeError: si ningún submodelo expone `get_feat`.
    """
    # Recorrido por todos los submodelos cargados. Se prefiere `.items()`
    # (en vez de `.values()`) por consistencia con el archivo original,
    # aunque la clave no se usa.
    for _, model in face_app.models.items():
        if hasattr(model, "get_feat"):
            return model

    # Mensaje explícito si nadie tiene `get_feat`: típicamente significa que
    # el `FaceAnalysis` se construyó con un nombre de modelo que no incluye
    # reconocimiento (raro en buffalo_*, posible en configs custom).
    raise RuntimeError(
        "No se encontró un modelo de reconocimiento con método get_feat()."
    )


# =========================================================
# 2) Re-extracción de embedding sobre una cara ya alineada
# =========================================================
# Toma el submodelo de reconocimiento + un rostro alineado y devuelve un
# vector 1D float32 con el embedding facial. El embedding crudo (sin L2)
# es lo que esperan las métricas de `phyloface.core.metrics` (que normalizan
# internamente cuando es necesario).
#
# Convenciones de entrada que asumimos:
#   - `aligned_rgb` ya pasó por alineación canónica y tiene la geometría
#     que el modelo espera (típicamente 112x112 RGB para buffalo_*).
#   - El dtype/rango lo maneja `get_feat` internamente (InsightFace
#     normaliza al rango que su backbone necesita).
# No depende de funciones propias del módulo.
def extract_embedding_from_aligned(rec_model, aligned_rgb: np.ndarray) -> np.ndarray:
    """
    Extrae el embedding facial desde una cara ya alineada.

    Parámetros:
        rec_model: submodelo de reconocimiento (ver `get_recognition_model`).
        aligned_rgb: imagen RGB del rostro ya alineado.

    Devuelve:
        ndarray float32 unidimensional con el embedding.
    """
    # === BUGFIX 2026-05-20 (ver cabecera del módulo) ===
    # `get_feat` usa internamente `cv2.dnn.blobFromImages(..., swapRB=True)`,
    # que asume input en BGR (default OpenCV) y lo convierte a RGB para el
    # modelo. Como nosotros tenemos RGB, debemos pasar BGR para que el swap
    # interno lo "devuelva" a RGB y el modelo reciba lo que esperaba en training.
    aligned_bgr = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)

    # `get_feat` puede devolver shape (1, D) o (D,) según implementación.
    # `flatten()` los unifica a (D,) sin copiar si el array ya es contiguo.
    emb = rec_model.get_feat(aligned_bgr).flatten()

    # Forzamos float32 explícito para mantener invariante de dtype con
    # el resto del paquete (`detector.py` también guarda embeddings float32).
    return emb.astype(np.float32)
