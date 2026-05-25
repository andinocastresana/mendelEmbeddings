# =========================================
# ID: PHYLOFACE_SMOKE_REGIONS_CANONICAL
# VERSION: v1.0
# =========================================
# Smoke test de Tareas #2/#3: contrato canonico de regiones.
#
# Verifica que el registry cargue, exponga los 12 nombres historicos de
# `regions_v2`, conserve simetrias izquierda/derecha y no contenga indices
# fuera del rango MediaPipe Face Mesh (0..477).

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phyloface.regions import (  # noqa: E402
    CANONICAL_REGIONS_VERSION,
    LANDMARKS_BACKEND,
    CANONICAL_REGION_NAMES,
    CANONICAL_REGION_SPECS,
    CANONICAL_REGION_BY_NAME,
    get_region_spec,
    paired_region_names,
)


EXPECTED_NAMES = (
    "left_eyebrow",
    "right_eyebrow",
    "left_eye",
    "right_eye",
    "left_cheekbone",
    "right_cheekbone",
    "left_cheek",
    "right_cheek",
    "nose",
    "mouth",
    "chin",
    "forehead",
)


def assert_indices_valid(name, kind, indices):
    if indices is None:
        return
    assert len(indices) >= 3, f"{name}.{kind} demasiado corto"
    assert all(isinstance(i, int) for i in indices), f"{name}.{kind} no-int"
    assert min(indices) >= 0, f"{name}.{kind} tiene indice negativo"
    assert max(indices) <= 477, f"{name}.{kind} fuera de Face Mesh 478"


def main():
    assert CANONICAL_REGIONS_VERSION == "regions-v2.0"
    assert LANDMARKS_BACKEND == "mediapipe-face-mesh-478"
    assert CANONICAL_REGION_NAMES == EXPECTED_NAMES
    assert len(CANONICAL_REGION_SPECS) == 12
    assert set(CANONICAL_REGION_BY_NAME) == set(EXPECTED_NAMES)

    for spec in CANONICAL_REGION_SPECS:
        assert get_region_spec(spec.name) is spec
        assert spec.extraction_modes == ("rect", "masked")
        assert spec.bbox_strategy in {"landmark_minmax", "chin_refined", "forehead_derived"}
        assert spec.mask_strategy in {"polygon", "convex_hull", "bbox_rect"}
        assert_indices_valid(spec.name, "landmark_idx", spec.landmark_idx)
        assert_indices_valid(spec.name, "polygon_idx", spec.polygon_idx)
        if spec.name == "forehead":
            assert spec.landmark_idx is None
            assert spec.mask_strategy == "bbox_rect"
        else:
            assert spec.landmark_idx is not None

    pairs = {tuple(sorted(p)) for p in paired_region_names()}
    assert pairs == {
        ("left_eyebrow", "right_eyebrow"),
        ("left_eye", "right_eye"),
        ("left_cheekbone", "right_cheekbone"),
        ("left_cheek", "right_cheek"),
    }

    for left, right in paired_region_names():
        assert get_region_spec(left).paired_with == right
        assert get_region_spec(right).paired_with == left

    print("[OK] regions canonical registry: 12 specs, indices, pairs, version")


if __name__ == "__main__":
    main()
