# =========================================
# ID: PHYLOFACE_VIZ_001
# VERSION: v1.0
# =========================================
# Origen de las funciones (migración Tarea #1, Paso 7):
#   src/phyloface_experimental_functions.py
#   - plot_regions_v2           (línea 839)
#   - plot_face_regions_overlay (línea 1070)
#   - plot_regions_v2_masked    (línea 1310)
#   - plot_region_detail        (línea 1365)
#
# Qué hace este módulo:
# - Visualizaciones del bloque de **regiones faciales** (módulo compañero:
#   `phyloface.regions`). Cuatro funciones complementarias:
#     plot_regions_v2          : grilla N×2 con los crops rectangulares
#                                de A y B lado a lado, una fila por región.
#                                Trabaja sobre el path "rect" de regiones.
#     plot_face_regions_overlay: superpone todas las máscaras de regiones
#                                sobre la cara completa, cada región con
#                                un color distintivo. Requiere regiones
#                                con clave `mask` (path "masked").
#     plot_regions_v2_masked   : grilla N×2 con uno de 3 modos
#                                ('rect', 'mask', 'masked') para inspeccionar
#                                cada faceta del path "masked".
#     plot_region_detail       : panel 1×4 para UNA región específica
#                                de UNA cara: overlay + crop + máscara
#                                + enmascarado. La vista más detallada.
#
# Decisiones de diseño:
# - El orden default de regiones agrupa por tipo anatómico
#   (cejas → ojos → pómulos → mejillas → centro → contorno), distinto
#   del orden alfabético — facilita la comparación visual A/B.
# - El `color_map` de `plot_face_regions_overlay` está hardcodeado con
#   colores discriminables. Si en el futuro se agregan regiones nuevas,
#   habrá que extenderlo.
# - Modo `masked` del `plot_regions_v2_masked` es el default porque es
#   el que mejor representa lo que el comparator_regional masked usa
#   como entrada.
# - Ninguna función devuelve `fig`; misma decisión que en
#   `detection.py` y `landmarks.py`.

# -----------------------------------------
# FILE: phyloface/viz/regions.py
# -----------------------------------------

import numpy as np
import matplotlib.pyplot as plt


# Orden default de regiones para los plots grid N×2.
# Agrupado por bloque anatómico: cejas, ojos, pómulos, mejillas,
# rasgos centrales, contorno. Más informativo que orden alfabético.
_DEFAULT_REGION_ORDER = [
    "left_eyebrow", "right_eyebrow",
    "left_eye", "right_eye",
    "left_cheekbone", "right_cheekbone",
    "left_cheek", "right_cheek",
    "nose", "mouth", "chin", "forehead",
]


# =========================================================
# 1) Grilla N×2 con crops rectangulares de cada región (A vs B)
# =========================================================
# Útil para inspección visual rápida del path "rect" (sin máscaras):
# ¿están razonablemente recortadas las regiones?
# Acepta `selected_pair` con `regions_v2` poblado por `add_regions_v2_to_pair`
# o por `add_regions_v2_masked_to_pair` (en el segundo caso solo se usa
# `crop_rgb`, ignorando `mask`/`crop_mask`/`crop_masked_rgb`).
# No depende de funciones propias del módulo.
def plot_regions_v2(selected_pair: dict, region_names: list[str] | None = None):
    """
    Grilla N×2 con los crops rectangulares de cada región para A y B.

    Parámetros:
        selected_pair: dict con `regions_v2` = {"A": {...}, "B": {...}}.
        region_names: lista opcional de regiones a mostrar; si None
            usa el orden default (12 regiones).
    """
    if region_names is None:
        region_names = list(_DEFAULT_REGION_ORDER)

    regions_a = selected_pair["regions_v2"]["A"]
    regions_b = selected_pair["regions_v2"]["B"]

    fig, axes = plt.subplots(len(region_names), 2, figsize=(6, 24))

    for i, region_name in enumerate(region_names):
        axes[i, 0].imshow(regions_a[region_name]["crop_rgb"])
        axes[i, 0].set_title(f"A - {region_name}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(regions_b[region_name]["crop_rgb"])
        axes[i, 1].set_title(f"B - {region_name}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()


# =========================================================
# 2) Overlay de todas las máscaras de regiones sobre la cara completa
# =========================================================
# Tipo "mapa de regiones": la cara de fondo, cada región pintada con
# un color distintivo translúcido. Útil para validar que las regiones
# cubren correctamente la cara y no se solapan demasiado.
# Requiere que las regiones tengan la clave `mask` (path "masked",
# producido por `add_regions_v2_masked_to_pair`). Regiones sin `mask`
# se saltan silenciosamente.
# No depende de funciones propias del módulo.
def plot_face_regions_overlay(
    image_rgb: np.ndarray,
    regions: dict,
    alpha: float = 0.30,
    title: str = "regions overlay",
):
    """
    Superpone las máscaras de las regiones sobre la cara completa.

    Parámetros:
        image_rgb: imagen RGB de la cara alineada (H, W, 3).
        regions: dict {region_name: dict_de_region} con clave 'mask'
            (típicamente `selected_pair["regions_v2"]["A"]`).
        alpha: opacidad de la mezcla (0=invisible, 1=color sólido).
        title: título del plot.
    """
    # Trabajamos sobre una copia para no mutar la imagen original.
    overlay = image_rgb.copy()

    # Mapa de colores discriminables por región. Si una región no
    # tiene color asignado (ej. región custom agregada después),
    # cae al gris por defecto.
    color_map = {
        "left_eye": (255, 0, 0),
        "right_eye": (0, 255, 0),
        "mouth": (255, 255, 0),
        "nose": (0, 0, 255),
        "left_eyebrow": (255, 0, 255),
        "right_eyebrow": (0, 255, 255),
        "left_cheekbone": (255, 128, 0),
        "right_cheekbone": (128, 255, 0),
        "left_cheek": (255, 128, 128),
        "right_cheek": (128, 255, 255),
        "chin": (180, 180, 255),
        "forehead": (200, 100, 255),
    }

    for region_name, region_data in regions.items():
        mask = region_data.get("mask")
        # Regiones que vinieron del path "rect" no tienen `mask` — las saltamos.
        if mask is None:
            continue

        color = np.array(color_map.get(region_name, (200, 200, 200)), dtype=np.float32)
        # Píxeles donde la máscara está activa: mezcla lineal entre
        # imagen original y color de la región (alpha controla el peso).
        idx = mask > 0
        overlay[idx] = ((1.0 - alpha) * overlay[idx] + alpha * color).astype(np.uint8)

    plt.figure(figsize=(5, 5))
    plt.imshow(overlay)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# =========================================================
# 3) Grilla N×2 con uno de tres modos (rect/mask/masked)
# =========================================================
# Análogo a `plot_regions_v2` pero permite elegir qué versión de cada
# región mostrar: el crop rectangular crudo, la máscara binaria, o el
# crop con fondo a 0 fuera de la máscara. El modo 'masked' es el
# default por ser el más informativo del path "masked".
# Requiere regiones del path "masked" (con `crop_rgb`, `crop_mask`,
# `crop_masked_rgb`).
# No depende de funciones propias del módulo.
def plot_regions_v2_masked(
    selected_pair: dict,
    region_names: list[str] | None = None,
    mode: str = "masked",
):
    """
    Grilla N×2 de regiones A vs B en uno de tres modos.

    Parámetros:
        selected_pair: dict con `regions_v2` poblado por
            `add_regions_v2_masked_to_pair`.
        region_names: lista opcional; si None usa el orden default.
        mode: 'rect' | 'mask' | 'masked' (default 'masked').
    """
    if region_names is None:
        region_names = list(_DEFAULT_REGION_ORDER)

    regions_a = selected_pair["regions_v2"]["A"]
    regions_b = selected_pair["regions_v2"]["B"]

    fig, axes = plt.subplots(len(region_names), 2, figsize=(6, 24))
    # Si len==1, plt.subplots devuelve Axes 1D (no 2D). Wrapping
    # defensivo para iterar igual con axes[i, j] abajo.
    if len(region_names) == 1:
        axes = np.array([axes])

    for i, region_name in enumerate(region_names):
        # Selección de cuál sub-imagen mostrar según modo.
        if mode == "rect":
            img_a = regions_a[region_name]["crop_rgb"]
            img_b = regions_b[region_name]["crop_rgb"]
            cmap = None
        elif mode == "mask":
            # Las máscaras son uint8 1-canal: cmap='gray' para visualizar
            # como blanco/negro.
            img_a = regions_a[region_name]["crop_mask"]
            img_b = regions_b[region_name]["crop_mask"]
            cmap = "gray"
        else:
            # Default 'masked': crop con fondo a 0 fuera de la máscara.
            img_a = regions_a[region_name]["crop_masked_rgb"]
            img_b = regions_b[region_name]["crop_masked_rgb"]
            cmap = None

        axes[i, 0].imshow(img_a, cmap=cmap)
        axes[i, 0].set_title(f"A - {region_name} [{mode}]")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(img_b, cmap=cmap)
        axes[i, 1].set_title(f"B - {region_name} [{mode}]")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()


# =========================================================
# 4) Panel 1×4 con todas las facetas de UNA región en UNA cara
# =========================================================
# La vista más detallada: para una región específica de una cara
# (A o B), muestra los 4 ángulos:
#   panel 0: overlay rojo de la máscara sobre la cara completa.
#   panel 1: crop_rgb rectangular.
#   panel 2: crop_mask binaria.
#   panel 3: crop_masked_rgb (con fondo a 0).
# Sirve para investigar a fondo qué pasa con una región particular
# cuando algo se ve raro en el grid agregado.
# No depende de funciones propias del módulo.
def plot_region_detail(
    selected_pair: dict,
    side: str,
    region_name: str,
):
    """
    Panel 1×4 con todas las facetas de una región en una cara.

    Parámetros:
        selected_pair: dict con `regions_v2` poblado por
            `add_regions_v2_masked_to_pair`.
        side: 'A' o 'B' (qué cara del par).
        region_name: nombre de la región (ej. 'left_eye').
    """
    if side not in ["A", "B"]:
        raise ValueError("side debe ser 'A' o 'B'")

    image_rgb = selected_pair["aligned_a"] if side == "A" else selected_pair["aligned_b"]
    region = selected_pair["regions_v2"][side][region_name]

    # Overlay sobre la cara completa: rojo translúcido (30% color +
    # 70% imagen original) en los píxeles dentro de la máscara.
    overlay = image_rgb.copy()
    mask = region["mask"]
    overlay[mask > 0] = (0.7 * overlay[mask > 0] + 0.3 * np.array([255, 0, 0])).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    axes[0].imshow(overlay)
    axes[0].set_title(f"{side} - {region_name}\noverlay")
    axes[0].axis("off")

    axes[1].imshow(region["crop_rgb"])
    axes[1].set_title("crop_rgb")
    axes[1].axis("off")

    axes[2].imshow(region["crop_mask"], cmap="gray")
    axes[2].set_title("crop_mask")
    axes[2].axis("off")

    axes[3].imshow(region["crop_masked_rgb"])
    axes[3].set_title("crop_masked_rgb")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()
