# =========================================
# ID: PHYLOFACE_SMOKE_007
# VERSION: v1.0
# =========================================
# Smoke test del Paso 7 (Tarea #1): subpaquete `phyloface.viz`.
#
# Verifica:
# - Imports de las 7 funciones nuevas vía re-export del __init__.py.
# - Cada función se ejecuta sin error sobre datos sintéticos.
# - No verifica el contenido visual (es smoke, no test de calidad).
#
# Ejecución:
#   python3 tests/smoke/test_paso_7_viz.py
#
# Backend: matplotlib 'Agg' (no-interactivo). Las llamadas a plt.show()
# son no-op gráfica pero ejecutan el rendering interno, suficiente
# para detectar errores en composición/shape/keys faltantes.

# -----------------------------------------
# FILE: tests/smoke/test_paso_7_viz.py
# -----------------------------------------

import sys
from pathlib import Path

# Backend no-interactivo ANTES de cualquier import de matplotlib o de
# módulos que importen matplotlib. Critical: tiene que ir antes que viz.*.
import matplotlib
matplotlib.use("Agg")

# Agregamos src/ al path para poder importar el paquete sin instalarlo.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt

from phyloface.viz import (
    plot_detected_faces,
    plot_face_triplet,
    plot_face_with_landmarks,
    plot_regions_v2,
    plot_face_regions_overlay,
    plot_regions_v2_masked,
    plot_region_detail,
)


# =========================================================
# Helpers de datos sintéticos
# =========================================================
def make_synthetic_face_record(face_id: str, photo_label: str, bbox, kps_global) -> dict:
    """Genera un face_record mínimo válido para plot_detected_faces / plot_face_triplet."""
    x1, y1, x2, y2 = bbox
    h, w = y2 - y1, x2 - x1
    rng = np.random.default_rng(hash(face_id) & 0xFFFFFFFF)
    crop_rgb = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return {
        "face_id": face_id,
        "photo_label": photo_label,
        "bbox": bbox,
        "bbox_raw": bbox,
        "det_score": 0.95,
        "crop_rgb": crop_rgb,
        "embedding": rng.normal(0, 1, size=(512,)).astype(np.float32),
        "kps": np.asarray(kps_global, dtype=np.float32),
    }


def make_synthetic_region(h: int, w: int, seed: int) -> dict:
    """Genera un dict de región completo (path 'masked')."""
    rng = np.random.default_rng(seed)
    crop_rgb = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    # Máscara global (sobre la imagen completa, no recortada). Para el
    # smoke alcanza con que tenga shape (H, W); le damos un disco simple.
    H, W = 224, 224
    yy, xx = np.ogrid[:H, :W]
    mask = ((yy - H//2)**2 + (xx - W//2)**2 < (min(H, W)//4)**2).astype(np.uint8) * 255
    crop_mask = (rng.integers(0, 2, size=(h, w)) * 255).astype(np.uint8)
    crop_masked_rgb = crop_rgb.copy()
    crop_masked_rgb[crop_mask == 0] = 0
    return {
        "bbox": (10, 10, 10 + w, 10 + h),
        "mask": mask,
        "crop_rgb": crop_rgb,
        "crop_mask": crop_mask,
        "crop_masked_rgb": crop_masked_rgb,
        "landmark_idx": None,
        "polygon_idx": None,
        "source": "approx",
    }


def make_synthetic_pair() -> dict:
    """selected_pair completo con regiones masked para los 4 plots de regions."""
    rng = np.random.default_rng(42)
    aligned_a = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    aligned_b = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)

    region_names = [
        "left_eyebrow", "right_eyebrow",
        "left_eye", "right_eye",
        "left_cheekbone", "right_cheekbone",
        "left_cheek", "right_cheek",
        "nose", "mouth", "chin", "forehead",
    ]
    # Tamaños arbitrarios por región (h, w distintos para chequear robustez)
    sizes = [(20, 40), (20, 40), (30, 50), (30, 50), (40, 50), (40, 50),
             (50, 60), (50, 60), (60, 40), (40, 70), (50, 80), (30, 100)]

    regions_a = {n: make_synthetic_region(h, w, seed=i)
                 for i, (n, (h, w)) in enumerate(zip(region_names, sizes))}
    regions_b = {n: make_synthetic_region(h, w, seed=i + 1000)
                 for i, (n, (h, w)) in enumerate(zip(region_names, sizes))}

    return {
        "aligned_a": aligned_a,
        "aligned_b": aligned_b,
        "regions_v2": {"A": regions_a, "B": regions_b},
    }


# =========================================================
# Smokes
# =========================================================
def smoke_detection():
    """plot_detected_faces + plot_face_triplet"""
    rng = np.random.default_rng(0)
    H, W = 400, 600
    img1 = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    img2 = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)

    rec1 = make_synthetic_face_record(
        "F1_R1", "F1", bbox=(100, 80, 250, 280),
        kps_global=[[140, 130], [210, 130], [175, 170], [150, 230], [200, 230]],
    )
    rec2 = make_synthetic_face_record(
        "F2_R1", "F2", bbox=(150, 100, 320, 320),
        kps_global=[[200, 160], [280, 160], [240, 200], [210, 270], [270, 270]],
    )

    # plot_detected_faces (2 fotos)
    plot_detected_faces({"F1": img1, "F2": img2}, [rec1, rec2])
    plt.close("all")

    # plot_detected_faces (1 sola foto: testea la rama `axes = [axes]`)
    plot_detected_faces({"F1": img1}, [rec1])
    plt.close("all")

    # plot_face_triplet: necesita el crop ya alineado
    aligned = rng.integers(0, 256, size=(112, 112, 3), dtype=np.uint8)
    plot_face_triplet(rec1, aligned)
    plt.close("all")
    print("[OK] viz.detection: plot_detected_faces (2 fotos + 1 foto) + plot_face_triplet")


def smoke_landmarks():
    """plot_face_with_landmarks"""
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    landmarks = rng.uniform(0, 224, size=(478, 2)).astype(np.float32)
    plot_face_with_landmarks(img, landmarks, title="smoke test landmarks")
    plt.close("all")
    print("[OK] viz.landmarks: plot_face_with_landmarks")


def smoke_regions():
    """plot_regions_v2 + plot_face_regions_overlay + plot_regions_v2_masked + plot_region_detail"""
    pair = make_synthetic_pair()

    # 1) Grid rect (default 12 regiones)
    plot_regions_v2(pair)
    plt.close("all")

    # 2) Overlay (color-mapped) de máscaras sobre cara completa
    plot_face_regions_overlay(pair["aligned_a"], pair["regions_v2"]["A"],
                              alpha=0.30, title="A overlay (smoke)")
    plt.close("all")

    # 3) Grid masked en los 3 modos
    for mode in ["rect", "mask", "masked"]:
        plot_regions_v2_masked(pair, mode=mode)
        plt.close("all")

    # 3b) Subset de regiones (testea la rama len(region_names)==1)
    plot_regions_v2_masked(pair, region_names=["left_eye"], mode="masked")
    plt.close("all")

    # 4) Detail (A y B, distintas regiones)
    plot_region_detail(pair, side="A", region_name="left_eye")
    plt.close("all")
    plot_region_detail(pair, side="B", region_name="mouth")
    plt.close("all")

    # side inválido -> ValueError
    try:
        plot_region_detail(pair, side="X", region_name="left_eye")
        assert False, "esperaba ValueError"
    except ValueError:
        pass

    print("[OK] viz.regions: plot_regions_v2 + plot_face_regions_overlay + "
          "plot_regions_v2_masked (3 modos + subset) + plot_region_detail + ValueError")


def main():
    print(f"[INFO] Backend matplotlib: {matplotlib.get_backend()}")
    smoke_detection()
    smoke_landmarks()
    smoke_regions()
    print()
    print("[OK] Paso 7 (viz/) verificado end-to-end con backend Agg.")


if __name__ == "__main__":
    main()
