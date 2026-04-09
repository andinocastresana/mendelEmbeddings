# =========================================
# ID: PHYLOFACE_HEATMAP_003
# VERSION: v1.0
# =========================================
# Reemplaza COMPLETO:
# phyloface/viz/heatmap.py

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np

from phyloface.core.models import DetectedFace


def add_face_thumbnail(ax, img, xy, zoom=0.32):
    """
    Inserta una miniatura en una posición del eje.
    """
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(
        imagebox,
        xy,
        xycoords="data",
        frameon=False,
        pad=0,
        annotation_clip=False,
    )
    ab.set_clip_on(False)
    ax.add_artist(ab)


def plot_similarity_heatmap(
    sim_matrix: np.ndarray,
    faces_a: list[DetectedFace],
    faces_b: list[DetectedFace],
    output_path=None,
    color_range_mode: str = "auto",   # "auto" | "fixed"
    vmin: float | None = None,
    vmax: float | None = None,
    file_label_a: str | None = None,
    file_label_b: str | None = None,
    score_label: str = "Score",
):
    """
    Dibuja la matriz de scores con miniaturas en filas y columnas.

    Parámetros:
    - color_range_mode:
        * "auto"  -> matplotlib decide el rango
        * "fixed" -> usa vmin / vmax
    - score_label:
        texto para colorbar y título principal
    """
    n_rows, n_cols = sim_matrix.shape

    # Estadísticos globales
    mean_score = float(np.mean(sim_matrix))
    median_score = float(np.median(sim_matrix))

    # Título y subtítulo
    title = f"{score_label} | media={mean_score:.3f} | mediana={median_score:.3f}"

    subtitle = None
    if file_label_a is not None and file_label_b is not None:
        subtitle = f"{file_label_a}  vs  {file_label_b}"

    fig, ax = plt.subplots(figsize=(2 + n_cols * 1.8, 2 + n_rows * 1.8))

    # Rango de color configurable
    if color_range_mode == "fixed":
        im = ax.imshow(sim_matrix, aspect="auto", vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(sim_matrix, aspect="auto")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(score_label)

    # Ticks sin etiquetas visibles
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels([""] * n_cols)
    ax.set_yticklabels([""] * n_rows)

    # Valores dentro de cada celda
    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(
                j,
                i,
                f"{sim_matrix[i, j]:.2f}",
                ha="center",
                va="center",
            )

    ax.set_xlabel("Rostros imagen B")
    ax.set_ylabel("Rostros imagen A")
    ax.set_title(title, pad=30)

    if subtitle is not None:
        ax.text(
            0.5,
            1.04,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
        )

    # Márgenes para miniaturas
    left_margin = 1.4
    top_margin = 1.4
    ax.set_xlim(-left_margin, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -top_margin)

    # Miniaturas arriba
    for j, face in enumerate(faces_b):
        add_face_thumbnail(ax, face.crop, (j, -1.0), zoom=0.35)

    # Miniaturas izquierda
    for i, face in enumerate(faces_a):
        add_face_thumbnail(ax, face.crop, (-1.0, i), zoom=0.35)

    # Rejilla fina
    ax.set_xticks([x - 0.5 for x in range(1, n_cols)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, n_rows)], minor=True)
    ax.grid(which="minor", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.subplots_adjust(left=0.22, top=0.82)

    if output_path is not None:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")

    plt.show()