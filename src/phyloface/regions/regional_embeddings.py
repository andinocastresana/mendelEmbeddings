# =========================================
# ID: PHYLOFACE_REGIONAL_EMBEDDINGS
# VERSION: v0.1
# =========================================
# Tarea #5: re-aplicar el backbone w600k_r50 a crops/máscaras regionales.
#
# Es un sanity layer: ArcFace fue entrenado para rostros completos alineados, no
# para parches de ojos/nariz/boca. Por eso este modulo expone calidad y validez
# por region, pero no promete que los embeddings regionales sean buenos.

from __future__ import annotations

import cv2
import numpy as np

from phyloface.core.embedder import extract_embedding_from_aligned
from phyloface.core.metrics import cosine_similarity
from phyloface.regions.canonical import CANONICAL_REGION_NAMES, CANONICAL_REGIONS_VERSION


REGIONAL_EMBEDDINGS_VERSION = f"{CANONICAL_REGIONS_VERSION}+arcface-crop-v0.1"


def _prepare_region_crop(crop_rgb: np.ndarray, output_size: tuple[int, int] = (112, 112)) -> np.ndarray:
    if crop_rgb.size == 0:
        raise ValueError("crop regional vacio")
    return cv2.resize(crop_rgb, output_size, interpolation=cv2.INTER_LINEAR)


def region_mask_fill_ratio(region: dict) -> float:
    crop_mask = region.get("crop_mask")
    if crop_mask is None:
        return 1.0
    if crop_mask.size == 0:
        return 0.0
    return float(np.count_nonzero(crop_mask) / crop_mask.size)


def extract_region_embeddings(
    rec_model,
    regions: dict,
    crop_key: str = "crop_masked_rgb",
    output_size: tuple[int, int] = (112, 112),
) -> dict[str, dict]:
    """
    Extrae embedding por region para un dict producido por extract_regions_v2_*.

    Devuelve region_name -> {embedding, valid, error, mask_fill_ratio}.
    """
    out: dict[str, dict] = {}
    for name in CANONICAL_REGION_NAMES:
        region = regions.get(name)
        if region is None:
            out[name] = {"embedding": None, "valid": False, "error": "missing_region", "mask_fill_ratio": 0.0}
            continue
        crop = region.get(crop_key) if crop_key in region else region.get("crop_rgb")
        try:
            prepared = _prepare_region_crop(crop, output_size=output_size)
            out[name] = {
                "embedding": extract_embedding_from_aligned(rec_model, prepared),
                "valid": True,
                "error": None,
                "mask_fill_ratio": region_mask_fill_ratio(region),
            }
        except Exception as exc:
            out[name] = {
                "embedding": None,
                "valid": False,
                "error": str(exc),
                "mask_fill_ratio": region_mask_fill_ratio(region),
            }
    return out


def compare_region_embeddings(emb_a: dict[str, dict], emb_b: dict[str, dict]) -> dict[str, dict]:
    """Coseno por region entre dos salidas de extract_region_embeddings."""
    scores: dict[str, dict] = {}
    for name in CANONICAL_REGION_NAMES:
        a = emb_a.get(name, {})
        b = emb_b.get(name, {})
        valid = bool(a.get("valid")) and bool(b.get("valid"))
        score = float("nan")
        if valid:
            score = float(cosine_similarity(a["embedding"], b["embedding"]))
        scores[name] = {
            "cosine": score,
            "valid": valid,
            "mask_fill_ratio_a": float(a.get("mask_fill_ratio", 0.0)),
            "mask_fill_ratio_b": float(b.get("mask_fill_ratio", 0.0)),
        }
    return scores


def region_embeddings_to_arrays(region_embeddings: dict[str, dict]) -> dict[str, np.ndarray]:
    """Convierte embeddings regionales a arrays compactos para cache .npz."""
    names = list(CANONICAL_REGION_NAMES)
    first = next((v["embedding"] for v in region_embeddings.values() if v.get("embedding") is not None), None)
    dim = int(first.shape[0]) if first is not None else 0
    embeddings = np.full((len(names), dim), np.nan, dtype=np.float32)
    valid = np.zeros((len(names),), dtype=np.bool_)
    fill = np.zeros((len(names),), dtype=np.float32)
    for i, name in enumerate(names):
        item = region_embeddings.get(name, {})
        fill[i] = float(item.get("mask_fill_ratio", 0.0))
        if item.get("valid") and item.get("embedding") is not None and dim:
            embeddings[i] = np.asarray(item["embedding"], dtype=np.float32)
            valid[i] = True
    return {
        "region_names": np.asarray(names, dtype="U32"),
        "region_embeddings": embeddings,
        "region_valid": valid,
        "region_mask_fill": fill,
    }
