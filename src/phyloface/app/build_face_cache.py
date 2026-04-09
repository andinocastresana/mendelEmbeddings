
# -----------------------------------------
# FILE: phyloface/app/build_face_cache.py
# -----------------------------------------
from pathlib import Path

from phyloface.core.config import (
    INPUT_DIR,
    VALID_EXTENSIONS,
    LIBRARY_NAME,
    MODEL_NAME,
    CTX_ID,
    DET_SIZE,
    FACE_SIZE,
    MAX_FACES,
)
from phyloface.core.detector import FaceDetector
from phyloface.core.cache import (
    make_config_dict,
    make_config_id,
    save_image_cache,
    get_cache_dir,
    inspect_one_cache,
)


def list_image_files(input_dir: Path, valid_extensions: set[str]) -> list[Path]:
    """
    Lista imágenes válidas de una carpeta.
    """
    return sorted(
        [
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_extensions
        ]
    )


def main():
    """
    Genera caches por imagen para la librería/modelo definidos en config.py
    """
    image_files = list_image_files(INPUT_DIR, VALID_EXTENSIONS)
    if not image_files:
        raise ValueError(f"No se encontraron imágenes en: {INPUT_DIR}")

    detector = FaceDetector(
        library_name=LIBRARY_NAME,
        model_name=MODEL_NAME,
        det_size=DET_SIZE,
        ctx_id=CTX_ID,
        max_faces=MAX_FACES,
    )

    config_dict = make_config_dict(
        library_name=LIBRARY_NAME,
        model_name=MODEL_NAME,
        det_size=DET_SIZE,
        face_size=FACE_SIZE,
        ctx_id=CTX_ID,
        max_faces=MAX_FACES,
    )
    config_id = make_config_id(config_dict)

    print(f"Imágenes encontradas: {len(image_files)}")
    print(f"Library   : {LIBRARY_NAME}")
    print(f"Model     : {MODEL_NAME}")
    print(f"Config ID : {config_id}\n")

    built_cache_dirs = []

    for image_path in image_files:
        print(f"[PROCESS] {image_path.name}")

        payload = detector.extract_faces_payload(
            image_path=image_path,
            face_size=FACE_SIZE,
        )

        meta_path, data_path = save_image_cache(
            payload=payload,
            config_dict=config_dict,
        )

        cache_dir = get_cache_dir(
            image_name=image_path.name,
            library_name=LIBRARY_NAME,
            model_name=MODEL_NAME,
            config_id=config_id,
        )
        built_cache_dirs.append(cache_dir)

        print(f"  n_faces   : {payload['n_faces']}")
        print(f"  meta.json : {meta_path}")
        print(f"  data.npz  : {data_path}\n")

    print("=" * 80)
    print("INSPECCIÓN RÁPIDA DEL PRIMER CACHE")
    print("=" * 80)

    first_summary = inspect_one_cache(built_cache_dirs[0])
    for k, v in first_summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()