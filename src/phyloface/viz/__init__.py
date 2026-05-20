# =========================================
# ID: PHYLOFACE_VIZ_001
# VERSION: v1.0
# =========================================
# Subpaquete: phyloface.viz
#
# Reúne todas las visualizaciones del proyecto. Organizado por dominio
# (qué se está visualizando), no por tipo de plot:
#
#   - detection.py : visualizaciones del bloque detección + alineación
#                    (panel multi-imagen con bboxes + face_id; panel 1x3
#                    de crop/keypoints/alineado de un rostro).
#   - landmarks.py : visualización de landmarks densos (imagen + scatter).
#   - regions.py   : visualizaciones de regiones faciales (grid N×2,
#                    overlay sobre cara completa, modos rect/mask/masked,
#                    detalle 1×4 de una región).
#   - heatmap.py   : heatmap matricial N×M de comparación pareada
#                    (compañero de `core.comparator`, NO de los módulos
#                    experimentales migrados en Tarea #1).
#
# Re-export plano para que el código cliente pueda hacer:
#     from phyloface.viz import plot_detected_faces, plot_face_with_landmarks, ...
# sin tener que conocer en qué archivo concreto vive cada función.

from phyloface.viz.detection import (
    plot_detected_faces,
    plot_face_triplet,
)
from phyloface.viz.landmarks import (
    plot_face_with_landmarks,
)
from phyloface.viz.regions import (
    plot_regions_v2,
    plot_face_regions_overlay,
    plot_regions_v2_masked,
    plot_region_detail,
)
# heatmap pre-existente del bundle PHYLOFACE_HEATMAP_003 (no migrado en Tarea #1).
from phyloface.viz.heatmap import (
    plot_similarity_heatmap,
    add_face_thumbnail,
)

__all__ = [
    # detection
    "plot_detected_faces",
    "plot_face_triplet",
    # landmarks
    "plot_face_with_landmarks",
    # regions
    "plot_regions_v2",
    "plot_face_regions_overlay",
    "plot_regions_v2_masked",
    "plot_region_detail",
    # heatmap (pre-existente)
    "plot_similarity_heatmap",
    "add_face_thumbnail",
]
