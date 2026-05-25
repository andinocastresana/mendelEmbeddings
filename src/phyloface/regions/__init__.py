# =========================================
# ID: PHYLOFACE_REGIONS_001
# VERSION: v1.0
# =========================================
# Subpaquete: phyloface.regions
#
# Reúne todo lo relacionado con la segmentación del rostro en regiones
# anatómicas (ojos, cejas, nariz, boca, mejillas, pómulos, mentón, frente)
# a partir de landmarks densos de MediaPipe Face Mesh.
#
# Hoy contiene dos backends complementarios para definir regiones:
#
#   - `geometry.py`     : constantes (índices de landmarks por región) +
#                         helpers de bbox y crop rectangular. Es la base.
#   - `extract_rect.py` : extrae regiones como **bbox rectangulares** + crop
#                         RGB. Es el camino "regiones v2".
#   - `extract_masked.py` (paso siguiente de la migración): mismas regiones
#                         pero con **máscara poligonal** además del rect.
#
# El módulo de visualización vive aparte en `phyloface.viz.regions`.
#
# Re-export selectivo: los usuarios deberían poder hacer:
#     from phyloface.regions import extract_regions_v2, add_regions_v2_to_pair
# sin tener que conocer en qué archivo concreto está cada cosa.

from phyloface.regions.canonical import (
    CANONICAL_REGIONS_VERSION,
    LANDMARKS_BACKEND,
    RegionSpec,
    CANONICAL_REGION_SPECS,
    CANONICAL_REGION_NAMES,
    CANONICAL_REGION_BY_NAME,
    get_region_spec,
    regions_for_group,
    paired_region_names,
)

from phyloface.regions.extract_rect import (
    extract_regions_v2,
    add_regions_v2_to_pair,
)

# Las constantes y helpers de geometry se exponen también, porque hay
# código (visualizaciones, futuras métricas geométricas Nivel A) que va a
# querer importarlos directamente sin pasar por extract_rect.
from phyloface.regions.geometry import (
    # Helpers
    connection_set_to_index_list,
    get_region_bbox,
    crop_from_bbox,
    get_forehead_bbox,
    get_chin_bbox_refined,
    # Constantes de índices por región (subconjuntos para bbox / convex hull)
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
    # Contornos poligonales ordenados (para fillPoly en máscaras)
    LEFT_EYE_POLYGON_IDX,
    RIGHT_EYE_POLYGON_IDX,
    MOUTH_POLYGON_IDX,
    LEFT_EYEBROW_POLYGON_IDX,
    RIGHT_EYEBROW_POLYGON_IDX,
    NOSE_POLYGON_IDX,
)

# Extractor de regiones con máscara poligonal + helpers asociados.
from phyloface.regions.extract_masked import (
    create_region_mask_from_points,
    crop_mask_and_image,
    extract_regions_v2_masked,
    add_regions_v2_masked_to_pair,
)
from phyloface.regions.geometric_features import (
    RegionGeometry,
    region_geometry,
    face_geometric_features,
    pair_geometric_differences,
)
from phyloface.regions.regional_embeddings import (
    REGIONAL_EMBEDDINGS_VERSION,
    extract_region_embeddings,
    compare_region_embeddings,
    region_embeddings_to_arrays,
    region_mask_fill_ratio,
)

__all__ = [
    # canonical registry
    "CANONICAL_REGIONS_VERSION",
    "LANDMARKS_BACKEND",
    "RegionSpec",
    "CANONICAL_REGION_SPECS",
    "CANONICAL_REGION_NAMES",
    "CANONICAL_REGION_BY_NAME",
    "get_region_spec",
    "regions_for_group",
    "paired_region_names",
    # extract_rect
    "extract_regions_v2",
    "add_regions_v2_to_pair",
    # extract_masked
    "create_region_mask_from_points",
    "crop_mask_and_image",
    "extract_regions_v2_masked",
    "add_regions_v2_masked_to_pair",
    # geometric features
    "RegionGeometry",
    "region_geometry",
    "face_geometric_features",
    "pair_geometric_differences",
    # regional embeddings
    "REGIONAL_EMBEDDINGS_VERSION",
    "extract_region_embeddings",
    "compare_region_embeddings",
    "region_embeddings_to_arrays",
    "region_mask_fill_ratio",
    # geometry helpers
    "connection_set_to_index_list",
    "get_region_bbox",
    "crop_from_bbox",
    "get_forehead_bbox",
    "get_chin_bbox_refined",
    # geometry constants — subconjuntos de índices
    "LEFT_EYE_IDX",
    "RIGHT_EYE_IDX",
    "LIPS_IDX",
    "NOSE_IDX",
    "LEFT_EYEBROW_IDX",
    "RIGHT_EYEBROW_IDX",
    "LEFT_CHEEKBONE_IDX",
    "RIGHT_CHEEKBONE_IDX",
    "LEFT_CHEEK_IDX",
    "RIGHT_CHEEK_IDX",
    "CHIN_IDX",
    # geometry constants — contornos poligonales
    "LEFT_EYE_POLYGON_IDX",
    "RIGHT_EYE_POLYGON_IDX",
    "MOUTH_POLYGON_IDX",
    "LEFT_EYEBROW_POLYGON_IDX",
    "RIGHT_EYEBROW_POLYGON_IDX",
    "NOSE_POLYGON_IDX",
]
