# =========================================
# ID: PHYLOFACE_REGIONS_CANONICAL
# VERSION: v1.0
# =========================================
# Contrato canonico de regiones faciales (Tareas #2 y #3).
#
# Este modulo NO extrae pixeles ni calcula metricas. Define la lista estable de
# regiones que el motor promete exponer hacia extractores, cache, comparadores y
# visualizaciones. La implementacion actual (`extract_rect.py` /
# `extract_masked.py`) sigue usando sus constantes historicas; este registry es
# la fuente de verdad nueva para que las tareas siguientes no dependan de dicts
# locales duplicados ni de nombres implícitos.
#
# Convenciones:
#   - `name` es estable y machine-readable. Cambiarlo rompe cache/UI.
#   - `landmark_idx` son indices MediaPipe Face Mesh usados para bbox o hull.
#   - `polygon_idx` es un contorno ordenado para mascara poligonal cuando existe.
#   - `bbox_strategy` documenta regiones con bbox especial (menton/frente).
#   - `mask_strategy` documenta como se obtiene la mascara en el path masked.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from phyloface.regions.geometry import (
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
    LEFT_EYE_POLYGON_IDX,
    RIGHT_EYE_POLYGON_IDX,
    MOUTH_POLYGON_IDX,
    LEFT_EYEBROW_POLYGON_IDX,
    RIGHT_EYEBROW_POLYGON_IDX,
    NOSE_POLYGON_IDX,
)

CANONICAL_REGIONS_VERSION = "regions-v2.0"
LANDMARKS_BACKEND = "mediapipe-face-mesh-478"

RegionSide = Literal["left", "right", "midline"]
RegionSource = Literal["mediapipe-official", "manual-approx", "derived-approx"]
ExtractionMode = Literal["rect", "masked"]
BBoxStrategy = Literal["landmark_minmax", "chin_refined", "forehead_derived"]
MaskStrategy = Literal["polygon", "convex_hull", "bbox_rect"]


@dataclass(frozen=True)
class RegionSpec:
    name: str
    label_es: str
    group: str
    side: RegionSide
    paired_with: str | None
    source: RegionSource
    landmark_idx: tuple[int, ...] | None
    polygon_idx: tuple[int, ...] | None
    default_pad: float | None
    bbox_strategy: BBoxStrategy
    mask_strategy: MaskStrategy
    extraction_modes: tuple[ExtractionMode, ...]
    notes: str = ""


def _t(xs: list[int] | tuple[int, ...] | None) -> tuple[int, ...] | None:
    return None if xs is None else tuple(int(x) for x in xs)


CANONICAL_REGION_SPECS: tuple[RegionSpec, ...] = (
    RegionSpec(
        name="left_eyebrow",
        label_es="ceja izquierda",
        group="eyebrow",
        side="left",
        paired_with="right_eyebrow",
        source="manual-approx",
        landmark_idx=_t(LEFT_EYEBROW_IDX),
        polygon_idx=_t(LEFT_EYEBROW_POLYGON_IDX),
        default_pad=0.20,
        bbox_strategy="landmark_minmax",
        mask_strategy="polygon",
        extraction_modes=("rect", "masked"),
        notes="Lista manual heredada; se interpreta como arco superciliar.",
    ),
    RegionSpec(
        name="right_eyebrow",
        label_es="ceja derecha",
        group="eyebrow",
        side="right",
        paired_with="left_eyebrow",
        source="manual-approx",
        landmark_idx=_t(RIGHT_EYEBROW_IDX),
        polygon_idx=_t(RIGHT_EYEBROW_POLYGON_IDX),
        default_pad=0.20,
        bbox_strategy="landmark_minmax",
        mask_strategy="polygon",
        extraction_modes=("rect", "masked"),
        notes="Lista manual heredada; se interpreta como arco superciliar.",
    ),
    RegionSpec(
        name="left_eye",
        label_es="ojo izquierdo",
        group="eye",
        side="left",
        paired_with="right_eye",
        source="mediapipe-official",
        landmark_idx=_t(LEFT_EYE_IDX),
        polygon_idx=_t(LEFT_EYE_POLYGON_IDX),
        default_pad=0.25,
        bbox_strategy="landmark_minmax",
        mask_strategy="polygon",
        extraction_modes=("rect", "masked"),
        notes="Landmarks de FACEMESH_LEFT_EYE; poligono reducido para parpado.",
    ),
    RegionSpec(
        name="right_eye",
        label_es="ojo derecho",
        group="eye",
        side="right",
        paired_with="left_eye",
        source="mediapipe-official",
        landmark_idx=_t(RIGHT_EYE_IDX),
        polygon_idx=_t(RIGHT_EYE_POLYGON_IDX),
        default_pad=0.25,
        bbox_strategy="landmark_minmax",
        mask_strategy="polygon",
        extraction_modes=("rect", "masked"),
        notes="Landmarks de FACEMESH_RIGHT_EYE; poligono reducido para parpado.",
    ),
    RegionSpec(
        name="left_cheekbone",
        label_es="pomulo izquierdo",
        group="cheekbone",
        side="left",
        paired_with="right_cheekbone",
        source="manual-approx",
        landmark_idx=_t(LEFT_CHEEKBONE_IDX),
        polygon_idx=None,
        default_pad=0.20,
        bbox_strategy="landmark_minmax",
        mask_strategy="convex_hull",
        extraction_modes=("rect", "masked"),
        notes="Subconjunto manual de la zona osea bajo el ojo.",
    ),
    RegionSpec(
        name="right_cheekbone",
        label_es="pomulo derecho",
        group="cheekbone",
        side="right",
        paired_with="left_cheekbone",
        source="manual-approx",
        landmark_idx=_t(RIGHT_CHEEKBONE_IDX),
        polygon_idx=None,
        default_pad=0.20,
        bbox_strategy="landmark_minmax",
        mask_strategy="convex_hull",
        extraction_modes=("rect", "masked"),
        notes="Subconjunto manual de la zona osea bajo el ojo.",
    ),
    RegionSpec(
        name="left_cheek",
        label_es="mejilla izquierda",
        group="cheek",
        side="left",
        paired_with="right_cheek",
        source="manual-approx",
        landmark_idx=_t(LEFT_CHEEK_IDX),
        polygon_idx=None,
        default_pad=0.20,
        bbox_strategy="landmark_minmax",
        mask_strategy="convex_hull",
        extraction_modes=("rect", "masked"),
        notes="Subconjunto manual de superficie media de la cara.",
    ),
    RegionSpec(
        name="right_cheek",
        label_es="mejilla derecha",
        group="cheek",
        side="right",
        paired_with="left_cheek",
        source="manual-approx",
        landmark_idx=_t(RIGHT_CHEEK_IDX),
        polygon_idx=None,
        default_pad=0.20,
        bbox_strategy="landmark_minmax",
        mask_strategy="convex_hull",
        extraction_modes=("rect", "masked"),
        notes="Subconjunto manual de superficie media de la cara.",
    ),
    RegionSpec(
        name="nose",
        label_es="nariz",
        group="nose",
        side="midline",
        paired_with=None,
        source="manual-approx",
        landmark_idx=_t(NOSE_IDX),
        polygon_idx=_t(NOSE_POLYGON_IDX),
        default_pad=0.22,
        bbox_strategy="landmark_minmax",
        mask_strategy="polygon",
        extraction_modes=("rect", "masked"),
        notes="Puente, alas y columela; contorno aproximado.",
    ),
    RegionSpec(
        name="mouth",
        label_es="boca",
        group="mouth",
        side="midline",
        paired_with=None,
        source="mediapipe-official",
        landmark_idx=_t(LIPS_IDX),
        polygon_idx=_t(MOUTH_POLYGON_IDX),
        default_pad=0.25,
        bbox_strategy="landmark_minmax",
        mask_strategy="polygon",
        extraction_modes=("rect", "masked"),
        notes="Landmarks de FACEMESH_LIPS; poligono externo de labios.",
    ),
    RegionSpec(
        name="chin",
        label_es="menton",
        group="chin",
        side="midline",
        paired_with=None,
        source="manual-approx",
        landmark_idx=_t(CHIN_IDX),
        polygon_idx=None,
        default_pad=None,
        bbox_strategy="chin_refined",
        mask_strategy="convex_hull",
        extraction_modes=("rect", "masked"),
        notes="Bbox refinada con labios como referencia para no invadir boca/cuello.",
    ),
    RegionSpec(
        name="forehead",
        label_es="frente",
        group="forehead",
        side="midline",
        paired_with=None,
        source="derived-approx",
        landmark_idx=None,
        polygon_idx=None,
        default_pad=None,
        bbox_strategy="forehead_derived",
        mask_strategy="bbox_rect",
        extraction_modes=("rect", "masked"),
        notes="Derivada desde cejas y ojos; Face Mesh no provee contorno cerrado.",
    ),
)

CANONICAL_REGION_NAMES: tuple[str, ...] = tuple(spec.name for spec in CANONICAL_REGION_SPECS)
CANONICAL_REGION_BY_NAME: dict[str, RegionSpec] = {
    spec.name: spec for spec in CANONICAL_REGION_SPECS
}


def get_region_spec(name: str) -> RegionSpec:
    """Devuelve la especificacion canonica de una region por nombre estable."""
    return CANONICAL_REGION_BY_NAME[name]


def regions_for_group(group: str) -> tuple[RegionSpec, ...]:
    """Filtra regiones por grupo anatomico (`eye`, `nose`, `cheek`, etc.)."""
    return tuple(spec for spec in CANONICAL_REGION_SPECS if spec.group == group)


def paired_region_names() -> tuple[tuple[str, str], ...]:
    """Pares izquierda/derecha canonicos, sin duplicar el inverso."""
    out = []
    seen = set()
    for spec in CANONICAL_REGION_SPECS:
        if spec.paired_with is None:
            continue
        key = tuple(sorted((spec.name, spec.paired_with)))
        if key not in seen:
            seen.add(key)
            out.append((spec.name, spec.paired_with))
    return tuple(out)
