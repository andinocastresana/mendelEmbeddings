# -----------------------------------------
# FILE: phyloface/app/run_pairwise_heatmap.py
# -----------------------------------------
from pathlib import Path

from phyloface.core.config import (
    MAX_FACES,
    DET_SIZE,
    CTX_ID,
    OUTPUT_DIR,
    COLOR_RANGE_MODE,
    VMIN,
    VMAX,
    METRIC,
    PAIR_JOBS,
)
from phyloface.core.detector import FaceDetector
from phyloface.core.comparator import FaceComparator
from phyloface.core.metrics import get_metric_label
from phyloface.viz.heatmap import plot_similarity_heatmap


def extract_simple_name(path: Path) -> str:
    """
    Extrae el texto tras el último '-' antes de la extensión.
    Si no hay '-', usa el stem completo.
    """
    stem = path.stem
    return stem.split("-")[-1] if "-" in stem else stem


def run_one_job(detector, comparator, job: dict):
    """
    Ejecuta un par de imágenes y guarda su heatmap.
    """
    image_a = Path(job["image_a"])
    image_b = Path(job["image_b"])
    output_path = OUTPUT_DIR / job["output_name"]

    faces_a = detector.detect_faces(image_a)
    faces_b = detector.detect_faces(image_b)

    score_matrix = comparator.compare_sets(
        faces_a=faces_a,
        faces_b=faces_b,
        metric=METRIC,
    )

    print("=" * 80)
    print(f"A: {image_a.name}")
    print(f"B: {image_b.name}")
    print(f"Métrica: {METRIC}")
    print("Shape matriz:", score_matrix.shape)
    print(score_matrix)

    plot_similarity_heatmap(
        sim_matrix=score_matrix,                      # se reutiliza el mismo argumento
        faces_a=faces_a,
        faces_b=faces_b,
        output_path=output_path,
        color_range_mode=COLOR_RANGE_MODE,
        vmin=VMIN,
        vmax=VMAX,
        file_label_a=extract_simple_name(image_a),
        file_label_b=extract_simple_name(image_b),
        score_label=get_metric_label(METRIC),        # requiere pequeño ajuste abajo
    )


def main():
    """
    Ejecuta todos los trabajos definidos en config.py
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detector = FaceDetector(
        det_size=DET_SIZE,
        ctx_id=CTX_ID,
        max_faces=MAX_FACES,
    )
    comparator = FaceComparator()

    for job in PAIR_JOBS:
        run_one_job(detector, comparator, job)


if __name__ == "__main__":
    main()


# -----------------------------------------
# AJUSTE NECESARIO EN phyloface/viz/heatmap.py
# -----------------------------------------
# Solo cambia la firma y dos líneas dentro de plot_similarity_heatmap:
#
# def plot_similarity_heatmap(
#     sim_matrix: np.ndarray,
#     faces_a: list[DetectedFace],
#     faces_b: list[DetectedFace],
#     output_path=None,
#     color_range_mode: str = "auto",
#     vmin: float | None = None,
#     vmax: float | None = None,
#     file_label_a: str | None = None,
#     file_label_b: str | None = None,
#     score_label: str = "Score",
# ):
#
# ...
# cbar.set_label(score_label)
# ...
# title = f"Comparación cruzada de rostros | media={mean_score:.3f} | mediana={median_score:.3f}"
#
# Si quieres, puedes cambiar ese title por:
# title = f"{score_label} | media={mean_score:.3f} | mediana={median_score:.3f}"