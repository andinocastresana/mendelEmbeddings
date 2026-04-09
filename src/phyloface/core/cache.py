# -----------------------------------------
# FILE: phyloface/core/cache.py
# -----------------------------------------
from pathlib import Path
import json
from datetime import datetime, timezone
import hashlib
import numpy as np

from phyloface.core.config import CACHE_ROOT, CACHE_SCHEMA_VERSION


def make_config_dict(
    library_name: str,
    model_name: str,
    det_size: tuple[int, int],
    face_size: tuple[int, int],
    ctx_id: int,
    max_faces: int,
) -> dict:
    """
    Diccionario de configuración del pipeline.
    """
    return {
        "library_name": library_name,
        "model_name": model_name,
        "det_size": list(det_size),
        "face_size": list(face_size),
        "ctx_id": int(ctx_id),
        "max_faces": int(max_faces),
        "cache_schema_version": CACHE_SCHEMA_VERSION,
    }


def make_config_id(config_dict: dict) -> str:
    """
    ID estable y legible para una configuración.
    """
    det_w, det_h = config_dict["det_size"]
    face_w, face_h = config_dict["face_size"]
    raw = json.dumps(config_dict, sort_keys=True)
    short_hash = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]

    return (
        f"det{det_w}x{det_h}"
        f"__face{face_w}x{face_h}"
        f"__max{config_dict['max_faces']}"
        f"__ctx{config_dict['ctx_id']}"
        f"__{short_hash}"
    )


def get_cache_dir(
    image_name: str,
    library_name: str,
    model_name: str,
    config_id: str,
) -> Path:
    """
    Ruta del cache de una imagen para una librería/modelo/config concretos.
    """
    return CACHE_ROOT / image_name / library_name / model_name / config_id


def save_image_cache(payload: dict, config_dict: dict) -> tuple[Path, Path]:
    """
    Guarda meta.json + data.npz.
    """
    image_name = payload["image_name"]
    library_name = config_dict["library_name"]
    model_name = config_dict["model_name"]
    config_id = make_config_id(config_dict)

    cache_dir = get_cache_dir(
        image_name=image_name,
        library_name=library_name,
        model_name=model_name,
        config_id=config_id,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta_path = cache_dir / "meta.json"
    data_path = cache_dir / "data.npz"

    meta = {
        "cache_created_at_utc": datetime.now(timezone.utc).isoformat(),
        "image": {
            "path": payload["image_path"],
            "name": payload["image_name"],
            "stem": payload["image_stem"],
            "suffix": payload["image_suffix"],
            "size_bytes": payload["image_size_bytes"],
            "mtime": payload["image_mtime"],
            "md5": payload["image_md5"],
            "width": payload["image_width"],
            "height": payload["image_height"],
        },
        "extractor": {
            "library_name": library_name,
            "model_name": model_name,
        },
        "config": config_dict,
        "config_id": config_id,
        "n_faces": payload["n_faces"],
        "arrays": {
            "indices": list(payload["indices"].shape),
            "bboxes": list(payload["bboxes"].shape),
            "det_scores": list(payload["det_scores"].shape),
            "embeddings": list(payload["embeddings"].shape),
            "crops": list(payload["crops"].shape),
            "kps": list(payload["kps"].shape),
            "landmark_3d_68": list(payload["landmark_3d_68"].shape),
            "gender": list(payload["gender"].shape),
            "age": list(payload["age"].shape),
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    np.savez_compressed(
        data_path,
        indices=payload["indices"],
        bboxes=payload["bboxes"],
        det_scores=payload["det_scores"],
        embeddings=payload["embeddings"],
        crops=payload["crops"],
        kps=payload["kps"],
        landmark_3d_68=payload["landmark_3d_68"],
        gender=payload["gender"],
        age=payload["age"],
    )

    return meta_path, data_path


def load_cache_meta(meta_path: Path) -> dict:
    """
    Lee el meta.json de un cache ya generado.
    """
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def inspect_one_cache(cache_dir: Path) -> dict:
    """
    Devuelve un resumen simple de un cache concreto.
    """
    meta_path = cache_dir / "meta.json"
    data_path = cache_dir / "data.npz"

    meta = load_cache_meta(meta_path)
    npz = np.load(data_path)

    return {
        "cache_dir": str(cache_dir),
        "image_name": meta["image"]["name"],
        "library_name": meta["extractor"]["library_name"],
        "model_name": meta["extractor"]["model_name"],
        "config_id": meta["config_id"],
        "n_faces": meta["n_faces"],
        "embedding_shape": tuple(npz["embeddings"].shape),
        "bbox_shape": tuple(npz["bboxes"].shape),
        "crop_shape": tuple(npz["crops"].shape),
    }

