# =========================================
# ID: PHYLOFACE_COMPARATOR_REGIONAL_001
# VERSION: v1.0
# =========================================
# Origen de las funciones (migración Tarea #1):
#   src/phyloface_experimental_functions.py
#   - resize_to_match                  (línea 877)
#   - grayscale_patch_cosine           (línea 890)
#   - compare_regions_v2               (línea 909)
#   - print_regional_summary           (línea 964)
#   - masked_grayscale_patch_cosine    (línea 1413)
#   - compare_regions_v2_masked        (línea 1435)
#
# Qué hace este módulo:
# - Calcula **scores regionales simples** entre las regiones de A y B de un
#   `selected_pair` ya procesado por `phyloface.regions`. Las métricas que
#   provee son **visuales y baratas** (gris + z-score + coseno entre vectores
#   planos): sirven como primer prototipo / baseline antes de pasar a
#   embeddings por región (Nivel B visual) o a features geométricos
#   (Nivel A) — esos llegan en tareas siguientes del proyecto.
#
# - Dos caminos paralelos:
#     compare_regions_v2          : usa el `crop_rgb` rectangular de cada región.
#                                   No tiene en cuenta la máscara.
#     compare_regions_v2_masked   : usa el `crop_masked_rgb` (fondo a 0 fuera
#                                   de la máscara). En la métrica ignora los
#                                   píxeles que están a 0 en ambos lados.
#
# - `print_regional_summary` imprime los scores de la última corrida para
#   inspección rápida en notebook / CLI.
#
# Limitación conocida (heredada del archivo original):
# - `print_regional_summary` asume la clave `gray_cosine` (la del path rect).
#   Si se ejecutó la versión masked, la clave es `gray_cosine_masked` y la
#   función no la imprime. No se cambia el comportamiento original en esta
#   migración para no introducir cambios funcionales encubiertos. Si se
#   quiere unificar más adelante, agregar un helper que detecte automáticamente
#   la clave disponible.
#
# Cosas que NO hace este módulo (a propósito):
# - No re-extrae regiones; espera que ya estén en `selected_pair["regions_v2"]`
#   con el formato que producen `phyloface.regions.extract_rect` o
#   `phyloface.regions.extract_masked`.
# - No comparte código con `phyloface.core.metrics` porque la métrica regional
#   trabaja sobre **parches enteros normalizados z-score**, no sobre vectores
#   de embedding L2-normalizados. Son productos distintos.

# -----------------------------------------
# FILE: phyloface/comparator_regional.py
# -----------------------------------------

import cv2
import numpy as np


# =========================================================
# 1) Helper: resize a tamaño común
# =========================================================
# Las regiones de A y B casi nunca tienen el mismo tamaño (bbox dependen
# del rostro y de la pose). Para una métrica vector-a-vector necesitamos
# que ambos parches tengan exactamente las mismas dimensiones.
# Interpolación lineal: barata y suficiente para una métrica de baseline.
# No depende de funciones propias del módulo.
def resize_to_match(img_a: np.ndarray, img_b: np.ndarray, size=(64, 64)):
    """
    Redimensiona dos parches al mismo tamaño con interpolación lineal.

    Parámetros:
        img_a, img_b: imágenes RGB (H, W, 3).
        size: (W, H) destino, default (64, 64).

    Devuelve:
        Tupla (a_resized, b_resized).
    """
    a = cv2.resize(img_a, size, interpolation=cv2.INTER_LINEAR)
    b = cv2.resize(img_b, size, interpolation=cv2.INTER_LINEAR)
    return a, b


# =========================================================
# 2) Métrica regional rectangular: grayscale + z-score + coseno
# =========================================================
# Calcula una métrica de similitud entre dos parches RGB del mismo tamaño:
#   1) pasa a escala de grises (descarta color: foco en estructura/intensidad).
#   2) aplana a vector.
#   3) z-score (resta media, divide std + epsilon) → robustez a iluminación.
#   4) coseno entre los vectores normalizados.
#   5) recorta a [-1, 1] por seguridad numérica.
# No depende de funciones propias del módulo.
def grayscale_patch_cosine(patch_a_rgb: np.ndarray, patch_b_rgb: np.ndarray) -> float:
    """
    Similitud coseno entre dos parches en gris con z-score.

    Parámetros:
        patch_a_rgb, patch_b_rgb: parches RGB del mismo tamaño.

    Devuelve:
        float en [-1, 1]. +1 = parches idénticos; 0 = no correlacionados.
    """
    # Conversión a gris + flatten. float32 para no perder precisión luego.
    gray_a = cv2.cvtColor(patch_a_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32).ravel()
    gray_b = cv2.cvtColor(patch_b_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32).ravel()

    # Z-score por parche: hace la métrica insensible a brillo y contraste
    # globales. 1e-8 evita división por 0 en parches uniformes.
    gray_a = (gray_a - gray_a.mean()) / (gray_a.std() + 1e-8)
    gray_b = (gray_b - gray_b.mean()) / (gray_b.std() + 1e-8)

    # Coseno: dot product normalizado por las normas L2 (que después del
    # z-score son aproximadamente sqrt(N), pero las recomputamos para
    # exactitud y por si los parches son muy uniformes).
    denom = (np.linalg.norm(gray_a) * np.linalg.norm(gray_b)) + 1e-8
    sim = float(np.dot(gray_a, gray_b) / denom)

    # Clipping defensivo: el coseno teórico está en [-1, 1], pero pequeñas
    # imprecisiones de punto flotante pueden devolver 1.0000001 o similar.
    return max(-1.0, min(1.0, sim))


# =========================================================
# 3) Métrica regional rectangular sobre todas las regiones
# =========================================================
# Itera sobre las regiones presentes en A y B (intersección), las pone al
# mismo tamaño y calcula `grayscale_patch_cosine` para cada una. Guarda los
# resultados en `selected_pair["regional_scores"]`.
#
# Regiones con crop vacío se saltan silenciosamente — puede pasar si la
# bbox quedó degenerada por landmarks raros.
#
# Depende de: resize_to_match, grayscale_patch_cosine.
def compare_regions_v2(selected_pair: dict, resize_shape=(64, 64)) -> dict:
    """
    Calcula la métrica regional simple (rect) sobre todas las regiones de A vs B.

    Parámetros:
        selected_pair: dict con 'regions_v2' = {"A": {...}, "B": {...}} ya
            poblado por `add_regions_v2_to_pair` o equivalente.
        resize_shape: (W, H) común para todos los parches antes de comparar.

    Devuelve:
        El mismo `selected_pair` con la clave 'regional_scores' agregada:
            { region_name: {"gray_cosine": float,
                            "shape_a": tuple,
                            "shape_b": tuple} }
    """
    regions_a = selected_pair["regions_v2"]["A"]
    regions_b = selected_pair["regions_v2"]["B"]

    # Intersección por nombre de región. sorted() para que el orden de
    # salida sea reproducible (útil para prints y comparaciones manuales).
    common_regions = sorted(set(regions_a.keys()).intersection(set(regions_b.keys())))
    regional_scores = {}

    for region_name in common_regions:
        crop_a = regions_a[region_name]["crop_rgb"]
        crop_b = regions_b[region_name]["crop_rgb"]

        # Skip silencioso si alguno de los crops está vacío. La alternativa
        # sería guardar NaN, pero la implementación original elige omitir.
        if crop_a.size == 0 or crop_b.size == 0:
            continue

        patch_a, patch_b = resize_to_match(crop_a, crop_b, size=resize_shape)
        regional_scores[region_name] = {
            "gray_cosine": grayscale_patch_cosine(patch_a, patch_b),
            "shape_a": crop_a.shape,
            "shape_b": crop_b.shape,
        }

    selected_pair["regional_scores"] = regional_scores
    return selected_pair


# =========================================================
# 4) Métrica regional enmascarada: ignora píxeles "fondo" en ambos lados
# =========================================================
# Variante de `grayscale_patch_cosine` pensada para parches que vienen del
# `crop_masked_rgb` (con fondo a 0 fuera de la máscara). Antes del z-score
# descarta los píxeles que están a 0 **en ambos** parches simultáneamente
# (interpretados como "fondo común"). Si después del filtrado no quedan
# píxeles válidos, devuelve NaN.
# No depende de funciones propias del módulo.
def masked_grayscale_patch_cosine(patch_a_rgb: np.ndarray, patch_b_rgb: np.ndarray) -> float:
    """
    Métrica coseno-en-gris ignorando píxeles que están a 0 en ambos parches.

    Parámetros:
        patch_a_rgb, patch_b_rgb: parches RGB del mismo tamaño, típicamente
            `crop_masked_rgb` con fondo a 0 fuera de la máscara.

    Devuelve:
        float en [-1, 1], o NaN si no queda ningún píxel válido.
    """
    gray_a = cv2.cvtColor(patch_a_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32).ravel()
    gray_b = cv2.cvtColor(patch_b_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32).ravel()

    # Mantenemos solo los píxeles que NO están a 0 simultáneamente en ambos.
    # No exigimos que estén a 0 en uno solo, porque un píxel oscuro real
    # (ej. iris negro) puede valer 0 sin ser fondo.
    valid = ~((gray_a == 0) & (gray_b == 0))
    gray_a = gray_a[valid]
    gray_b = gray_b[valid]

    # Si la intersección de áreas válidas quedó vacía, no hay métrica posible.
    # NaN es la convención del archivo original; el llamador decide qué hacer.
    if len(gray_a) == 0 or len(gray_b) == 0:
        return np.nan

    # Z-score sobre el subconjunto válido y coseno (idéntico al path rect).
    gray_a = (gray_a - gray_a.mean()) / (gray_a.std() + 1e-8)
    gray_b = (gray_b - gray_b.mean()) / (gray_b.std() + 1e-8)

    denom = (np.linalg.norm(gray_a) * np.linalg.norm(gray_b)) + 1e-8
    sim = float(np.dot(gray_a, gray_b) / denom)
    return max(-1.0, min(1.0, sim))


# =========================================================
# 5) Métrica regional enmascarada sobre todas las regiones
# =========================================================
# Análoga a `compare_regions_v2` pero leyendo `crop_masked_rgb` y usando
# `masked_grayscale_patch_cosine`. Llave de salida: `gray_cosine_masked`
# (distinta del path rect: a propósito, para no confundirlas).
#
# Nota: usa `cv2.resize` directamente (en vez de `resize_to_match`) para
# preservar exactamente el flujo del archivo original; funcionalmente
# equivalente.
#
# Depende de: masked_grayscale_patch_cosine.
def compare_regions_v2_masked(selected_pair: dict, resize_shape=(64, 64)) -> dict:
    """
    Calcula la métrica regional enmascarada sobre todas las regiones de A vs B.

    Parámetros:
        selected_pair: dict con 'regions_v2' = {"A": ..., "B": ...} en formato
            enmascarado (cada región con `crop_masked_rgb`).
        resize_shape: (W, H) común para todos los parches antes de comparar.

    Devuelve:
        El mismo `selected_pair` con 'regional_scores':
            { region_name: {"gray_cosine_masked": float | NaN,
                            "shape_a": tuple,
                            "shape_b": tuple} }
    """
    regions_a = selected_pair["regions_v2"]["A"]
    regions_b = selected_pair["regions_v2"]["B"]

    common_regions = sorted(set(regions_a.keys()).intersection(set(regions_b.keys())))
    regional_scores = {}

    for region_name in common_regions:
        crop_a = regions_a[region_name]["crop_masked_rgb"]
        crop_b = regions_b[region_name]["crop_masked_rgb"]

        if crop_a.size == 0 or crop_b.size == 0:
            continue

        # cv2.resize directo (no usamos resize_to_match) para reproducir
        # exactamente el orden y el flujo del archivo original.
        patch_a = cv2.resize(crop_a, resize_shape, interpolation=cv2.INTER_LINEAR)
        patch_b = cv2.resize(crop_b, resize_shape, interpolation=cv2.INTER_LINEAR)

        regional_scores[region_name] = {
            "gray_cosine_masked": masked_grayscale_patch_cosine(patch_a, patch_b),
            "shape_a": crop_a.shape,
            "shape_b": crop_b.shape,
        }

    selected_pair["regional_scores"] = regional_scores
    return selected_pair


# =========================================================
# 6) Print de inspección rápida
# =========================================================
# Imprime el dict `regional_scores` ordenado por nombre de región, mostrando
# la clave `gray_cosine` por entrada.
#
# Limitación heredada del archivo original (ver cabecera del módulo):
#   - Sólo entiende la clave `gray_cosine` (path rect). Si el `selected_pair`
#     viene del path masked, la clave es `gray_cosine_masked` y este print
#     fallará. No se cambia para no introducir cambios funcionales encubiertos.
# No depende de funciones propias del módulo.
def print_regional_summary(selected_pair: dict):
    """
    Resumen corto del dict `regional_scores` (path rect).
    """
    print("=== COMPARACIÓN REGIONAL ===")
    for region_name, metrics in sorted(selected_pair["regional_scores"].items()):
        print(f"{region_name:16s} | gray_cosine={metrics['gray_cosine']:.4f}")
