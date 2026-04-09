# -----------------------------------------
# FILE: phyloface/app/run_folder_batch.py
# -----------------------------------------
from itertools import combinations
from pathlib import Path

import pandas as pd
import numpy as np

from phyloface.core.config import (
    MAX_FACES,
    DET_SIZE,
    CTX_ID,
    INPUT_DIR,
    OUTPUT_DIR,
    VALID_EXTENSIONS,
    METRICS,
    DETAIL_CSV,
    SUMMARY_CSV,
    DETAIL_PARQUET,
    SUMMARY_PARQUET,
)
from phyloface.core.detector import FaceDetector
from phyloface.core.metrics import get_metric_function


def extract_simple_name(path: Path) -> str:
    """
    Devuelve el texto entre el último '-' y la extensión.
    Si no hay '-', devuelve el stem completo.
    """
    stem = path.stem
    return stem.split("-")[-1] if "-" in stem else stem


def list_image_files(input_dir: Path, valid_extensions: set[str]) -> list[Path]:
    """
    Lista imágenes válidas dentro de una carpeta.
    """
    files = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_extensions
    ]
    return sorted(files)


def build_metric_registry(metric_names: list[str]) -> dict[str, callable]:
    """
    Construye un diccionario nombre -> función de métrica.
    """
    return {name: get_metric_function(name) for name in metric_names}


def compare_faces_multi_metric(
    faces_a,
    faces_b,
    metric_registry: dict[str, callable],
) -> list[dict]:
    """
    Genera una fila por comparación rostro_a x rostro_b,
    calculando todas las métricas solicitadas.
    """
    rows = []

    for i, face_a in enumerate(faces_a):
        for j, face_b in enumerate(faces_b):
            row = {
                "face_idx_a": i,
                "face_idx_b": j,
                "bbox_a": tuple(face_a.bbox),
                "bbox_b": tuple(face_b.bbox),
            }

            for metric_name, metric_fn in metric_registry.items():
                row[metric_name] = float(
                    metric_fn(face_a.embedding, face_b.embedding)
                )

            rows.append(row)

    return rows


def summarize_pair(df_detail_pair: pd.DataFrame, pair_meta: dict, metric_names: list[str]) -> dict:
    """
    Resume un par de imágenes con estadísticos globales por métrica.
    """
    summary = {
        **pair_meta,
        "n_faces_a": int(df_detail_pair["face_idx_a"].nunique()),
        "n_faces_b": int(df_detail_pair["face_idx_b"].nunique()),
        "n_comparisons": int(len(df_detail_pair)),
    }

    for metric in metric_names:
        values = df_detail_pair[metric].astype(float).to_numpy()
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_median"] = float(np.median(values))
        summary[f"{metric}_std"] = float(np.std(values))
        summary[f"{metric}_min"] = float(np.min(values))
        summary[f"{metric}_max"] = float(np.max(values))

    return summary


def main():
    """
    Recorre todos los archivos de INPUT_DIR y compara todos contra todos.
    Guarda:
    - detalle por rostro x rostro
    - resumen por imagen x imagen
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_files = list_image_files(INPUT_DIR, VALID_EXTENSIONS)
    if len(image_files) < 2:
        raise ValueError(f"Se necesitan al menos 2 imágenes en: {INPUT_DIR}")

    print(f"Imágenes detectadas: {len(image_files)}")
    for p in image_files:
        print(" -", p.name)

    detector = FaceDetector(
        det_size=DET_SIZE,
        ctx_id=CTX_ID,
        max_faces=MAX_FACES,
    )

    metric_registry = build_metric_registry(METRICS)

    # Cachear detecciones para no recalcular si una imagen participa en varios pares
    detected_faces_by_image: dict[Path, list] = {}

    for image_path in image_files:
        detected_faces_by_image[image_path] = detector.detect_faces(image_path)
        print(
            f"[OK] {image_path.name} -> "
            f"{len(detected_faces_by_image[image_path])} rostros detectados"
        )

    detail_rows = []
    summary_rows = []

    # Todos contra todos, sin repetir y sin diagonal
    for image_a, image_b in combinations(image_files, 2):
        faces_a = detected_faces_by_image[image_a]
        faces_b = detected_faces_by_image[image_b]

        pair_meta = {
            "image_a_path": str(image_a),
            "image_b_path": str(image_b),
            "image_a_name": image_a.name,
            "image_b_name": image_b.name,
            "image_a_label": extract_simple_name(image_a),
            "image_b_label": extract_simple_name(image_b),
        }

        pair_rows = compare_faces_multi_metric(
            faces_a=faces_a,
            faces_b=faces_b,
            metric_registry=metric_registry,
        )

        # Añadir metadatos del par a cada fila de detalle
        for row in pair_rows:
            row.update(pair_meta)

        df_pair_detail = pd.DataFrame(pair_rows)
        detail_rows.extend(df_pair_detail.to_dict(orient="records"))

        summary_rows.append(
            summarize_pair(
                df_detail_pair=df_pair_detail,
                pair_meta=pair_meta,
                metric_names=METRICS,
            )
        )

        print(
            f"[PAIR] {image_a.name} vs {image_b.name} -> "
            f"{len(faces_a)}x{len(faces_b)} = {len(df_pair_detail)} comparaciones"
        )

    # Guardado final
    df_detail = pd.DataFrame(detail_rows)
    df_summary = pd.DataFrame(summary_rows)

    df_detail.to_csv(DETAIL_CSV, index=False)
    df_summary.to_csv(SUMMARY_CSV, index=False)

    # Intentar guardar parquet si el entorno tiene pyarrow o fastparquet
    try:
        df_detail.to_parquet(DETAIL_PARQUET, index=False)
        df_summary.to_parquet(SUMMARY_PARQUET, index=False)
        parquet_msg = "CSV + Parquet"
    except Exception:
        parquet_msg = "solo CSV"

    print("\nGuardado completado:")
    print(" -", DETAIL_CSV)
    print(" -", SUMMARY_CSV)
    print("Formato adicional:", parquet_msg)


if __name__ == "__main__":
    main()