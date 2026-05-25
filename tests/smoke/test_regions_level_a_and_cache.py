import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phyloface.core.cache import make_config_dict, make_config_id  # noqa: E402
from phyloface.regions import (  # noqa: E402
    CANONICAL_REGION_NAMES,
    CANONICAL_REGIONS_VERSION,
    face_geometric_features,
    pair_geometric_differences,
    region_embeddings_to_arrays,
)


def synthetic_landmarks(offset_x: float = 0.0) -> np.ndarray:
    pts = np.zeros((478, 2), dtype=np.float32)
    for i in range(478):
        pts[i] = [20.0 + (i % 40) * 1.6 + offset_x, 20.0 + (i // 40) * 3.2]
    return pts


def test_geometric_features_and_cache_contract():
    landmarks_a = synthetic_landmarks()
    landmarks_b = synthetic_landmarks(offset_x=1.5)
    image_shape = (112, 112, 3)

    features = face_geometric_features(landmarks_a, image_shape)
    diffs = pair_geometric_differences(landmarks_a, landmarks_b, image_shape, image_shape)
    assert "dist_nose_to_mouth" in features
    assert "sym_left_eye_right_eye_width_absdiff" in features
    assert "geom_absdiff_dist_nose_to_mouth" in diffs
    assert all(np.isfinite(v) for v in features.values())

    base = make_config_dict("insightface", "buffalo_l", (640, 640), (112, 112), -1, 1)
    regional = make_config_dict(
        "insightface",
        "buffalo_l",
        (640, 640),
        (112, 112),
        -1,
        1,
        regions_version=CANONICAL_REGIONS_VERSION,
        region_extraction_mode="masked",
        region_embedding_model="w600k_r50",
    )
    assert make_config_id(base) != make_config_id(regional)
    assert CANONICAL_REGIONS_VERSION in make_config_id(regional)
    assert "masked" in make_config_id(regional)

    region_embeddings = {
        name: {
            "embedding": np.ones((512,), dtype=np.float32) * i,
            "valid": True,
            "mask_fill_ratio": 0.5,
        }
        for i, name in enumerate(CANONICAL_REGION_NAMES)
    }
    arrays = region_embeddings_to_arrays(region_embeddings)
    assert arrays["region_embeddings"].shape == (len(CANONICAL_REGION_NAMES), 512)
    assert arrays["region_valid"].all()


if __name__ == "__main__":
    test_geometric_features_and_cache_contract()
    print("[OK] regions level A features + cache regional contract")
