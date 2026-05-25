# =========================================
# ID: PHYLOFACE_REGIONS_GEOMETRIC_FEATURES
# VERSION: v0.1
# =========================================
# Tarea #4: features geometricas Nivel A sobre landmarks MediaPipe.
#
# Este modulo no extrae pixeles ni embeddings. Convierte landmarks densos en un
# vector/dict compacto de distancias, proporciones, angulos y simetrias. Las
# magnitudes se normalizan por distancia interpupilar para que sean comparables
# entre caras alineadas de distinto tamano.

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from phyloface.regions.canonical import (
    CANONICAL_REGION_SPECS,
    CANONICAL_REGION_NAMES,
    paired_region_names,
)
from phyloface.regions.geometry import get_region_bbox, get_chin_bbox_refined, get_forehead_bbox


@dataclass(frozen=True)
class RegionGeometry:
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    width: float
    height: float
    area: float


def _centroid(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float32).mean(axis=0)


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def _angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    delta = np.asarray(b, dtype=np.float32) - np.asarray(a, dtype=np.float32)
    return float(math.degrees(math.atan2(float(delta[1]), float(delta[0]))))


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if abs(den) > 1e-8 else float("nan")


def region_geometry(landmarks: np.ndarray, image_shape: tuple) -> dict[str, RegionGeometry]:
    """Calcula bbox, centroide y tamanos por region canonica."""
    out: dict[str, RegionGeometry] = {}
    for spec in CANONICAL_REGION_SPECS:
        if spec.bbox_strategy == "forehead_derived":
            bbox = get_forehead_bbox(landmarks, image_shape)
            pts = landmarks[list(spec.polygon_idx or spec.landmark_idx or [])] if spec.landmark_idx else None
            centroid = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0], dtype=np.float32)
        elif spec.bbox_strategy == "chin_refined":
            mouth = next(s for s in CANONICAL_REGION_SPECS if s.name == "mouth")
            bbox = get_chin_bbox_refined(
                landmarks=landmarks,
                image_shape=image_shape,
                chin_idx=list(spec.landmark_idx or ()),
                lips_idx=list(mouth.landmark_idx or ()),
            )
            pts = landmarks[list(spec.landmark_idx or ())]
            centroid = _centroid(pts)
        else:
            idx = list(spec.landmark_idx or ())
            bbox = get_region_bbox(landmarks, idx, image_shape, pad=spec.default_pad or 0.20)
            pts = landmarks[idx]
            centroid = _centroid(pts)

        x1, y1, x2, y2 = bbox
        width = float(max(0, x2 - x1))
        height = float(max(0, y2 - y1))
        out[spec.name] = RegionGeometry(
            bbox=bbox,
            centroid=(float(centroid[0]), float(centroid[1])),
            width=width,
            height=height,
            area=float(width * height),
        )
    return out


def face_geometric_features(landmarks: np.ndarray, image_shape: tuple) -> dict[str, float]:
    """Devuelve features geometricas normalizadas para una cara."""
    geom = region_geometry(landmarks, image_shape)
    centers = {name: np.array(geom[name].centroid, dtype=np.float32) for name in CANONICAL_REGION_NAMES}

    left_eye = centers["left_eye"]
    right_eye = centers["right_eye"]
    mouth = centers["mouth"]
    nose = centers["nose"]
    chin = centers["chin"]
    forehead = centers["forehead"]
    scale = max(_euclidean(left_eye, right_eye), 1.0)

    features: dict[str, float] = {
        "scale_interocular": scale,
        "angle_eye_axis_deg": _angle_degrees(left_eye, right_eye),
        "dist_eye_to_eye": _safe_ratio(_euclidean(left_eye, right_eye), scale),
        "dist_nose_to_mouth": _safe_ratio(_euclidean(nose, mouth), scale),
        "dist_nose_to_chin": _safe_ratio(_euclidean(nose, chin), scale),
        "dist_mouth_to_chin": _safe_ratio(_euclidean(mouth, chin), scale),
        "dist_forehead_to_nose": _safe_ratio(_euclidean(forehead, nose), scale),
    }

    for name in CANONICAL_REGION_NAMES:
        g = geom[name]
        features[f"{name}_width_ratio"] = _safe_ratio(g.width, scale)
        features[f"{name}_height_ratio"] = _safe_ratio(g.height, scale)
        features[f"{name}_area_ratio"] = _safe_ratio(g.area, scale * scale)

    for left, right in paired_region_names():
        gl = geom[left]
        gr = geom[right]
        features[f"sym_{left}_{right}_width_absdiff"] = _safe_ratio(abs(gl.width - gr.width), scale)
        features[f"sym_{left}_{right}_height_absdiff"] = _safe_ratio(abs(gl.height - gr.height), scale)
        features[f"sym_{left}_{right}_area_reldiff"] = _safe_ratio(abs(gl.area - gr.area), max(gl.area, gr.area, 1.0))
        features[f"sym_{left}_{right}_centroid_y_absdiff"] = _safe_ratio(abs(gl.centroid[1] - gr.centroid[1]), scale)

    return features


def pair_geometric_differences(
    landmarks_a: np.ndarray,
    landmarks_b: np.ndarray,
    image_shape_a: tuple,
    image_shape_b: tuple,
) -> dict[str, float]:
    """Diferencias absolutas feature-a-feature entre dos caras."""
    fa = face_geometric_features(landmarks_a, image_shape_a)
    fb = face_geometric_features(landmarks_b, image_shape_b)
    keys = sorted(set(fa).intersection(fb))
    return {f"geom_absdiff_{k}": float(abs(fa[k] - fb[k])) for k in keys}
