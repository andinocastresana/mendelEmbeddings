from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from insightface.app import FaceAnalysis


# =========================
# Config
# =========================
MAX_FACES = 11
DET_SIZE = (640, 640)
CTX_ID = -1   # CPU; usa 0 si tienes GPU configurada
FACE_SIZE = (112, 112)


# =========================
# Core class
# =========================
class MultiFaceComparator:
    def __init__(self, det_size=(640, 640), ctx_id=-1, max_faces=11):
        self.max_faces = max_faces
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def detect_faces(self, image_path):
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"No se pudo leer la imagen: {image_path}")

        faces = self.app.get(img)

        if len(faces) == 0:
            raise ValueError(f"No se detectaron caras en: {image_path}")

        # ordenar de izquierda a derecha usando x1 del bbox
        faces = sorted(faces, key=lambda f: f.bbox[0])

        # limitar a max_faces
        faces = faces[:self.max_faces]

        results = []
        h, w = img.shape[:2]

        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox.astype(int)

            # asegurar límites válidos
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_resized = cv2.resize(crop_rgb, FACE_SIZE)

            results.append({
                "index": i,
                "bbox": (x1, y1, x2, y2),
                "crop": crop_resized,
                "embedding": face.embedding
            })

        if len(results) == 0:
            raise ValueError(f"Se detectaron caras pero no se pudieron recortar correctamente en: {image_path}")

        return results

    @staticmethod
    def cosine_similarity(vec1, vec2):
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        return float(np.dot(vec1, vec2))

    def compare_sets(self, faces_a, faces_b):
        sim_matrix = np.zeros((len(faces_a), len(faces_b)), dtype=float)

        for i, fa in enumerate(faces_a):
            for j, fb in enumerate(faces_b):
                sim_matrix[i, j] = self.cosine_similarity(
                    fa["embedding"],
                    fb["embedding"]
                )

        return sim_matrix


# =========================
# Plot helper
# =========================
def add_image_to_axis(ax, img, xy, zoom=0.45):
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(
        imagebox,
        xy,
        frameon=False,
        xycoords="data",
        boxcoords="offset points",
        pad=0
    )
    ax.add_artist(ab)


def add_face_thumbnail(ax, img, xy, zoom=0.32):
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(
        imagebox,
        xy,
        xycoords="data",
        frameon=False,
        pad=0,
        annotation_clip=False
    )
    ab.set_clip_on(False)
    ax.add_artist(ab)


def plot_similarity_heatmap(sim_matrix, faces_a, faces_b, output_path=None):
    n_rows, n_cols = sim_matrix.shape

    fig, ax = plt.subplots(figsize=(2 + n_cols * 1.8, 2 + n_rows * 1.8))

    im = ax.imshow(sim_matrix, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity")

    # ticks
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels([""] * n_cols)
    ax.set_yticklabels([""] * n_rows)

    # valores en celdas
    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(
                j, i,
                f"{sim_matrix[i, j]:.2f}",
                ha="center",
                va="center"
            )

    ax.set_xlabel("Rostros imagen B")
    ax.set_ylabel("Rostros imagen A")
    ax.set_title("Comparación cruzada de rostros")

    # ampliar límites para dejar espacio visible a miniaturas
    left_margin = 1.4
    top_margin = 1.4

    ax.set_xlim(-left_margin, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -top_margin)

    # miniaturas columnas (arriba)
    for j, face in enumerate(faces_b):
        add_face_thumbnail(ax, face["crop"], (j, -1.0), zoom=0.35)

    # miniaturas filas (izquierda)
    for i, face in enumerate(faces_a):
        add_face_thumbnail(ax, face["crop"], (-1.0, i), zoom=0.35)

    # rejilla opcional para leer mejor
    ax.set_xticks([x - 0.5 for x in range(1, n_cols)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, n_rows)], minor=True)
    ax.grid(which="minor", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # más margen alrededor de la figura
    plt.subplots_adjust(left=0.22, top=0.82)

    if output_path is not None:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")

    plt.show()
# =========================
# Main
# =========================
if __name__ == "__main__":
    #image_a = Path("data/input/img/IMG_20260104_144832707.jpg")
    #image_a = Path("data/input/img/fraternos_jovenes.jpg")
    #image_b = Path("data/input/img/fraternosChacabuco8.jpg")
    #image_b = Path("data/input/img/fraternosAsado.jpg")
    #image_b = Path("data/input/img/Señores_o_gente.jpg")
    #image_b =Path("data/input/img/brunoRio.jpg")
    #image_b =Path("data/input/img/DiegoSofiaY3Mas.jpg")
    image_a = Path("data/input/img/teams/seleccion-de-argelia.jpg")
    image_b = Path("data/input/img/teams/seleccion-de-argentina.jpg")
    output_plot = Path("data/output/heatmap_faces.png")

    output_plot.parent.mkdir(parents=True, exist_ok=True)

    comparator = MultiFaceComparator(
        det_size=DET_SIZE,
        ctx_id=CTX_ID,
        max_faces=MAX_FACES
    )

    faces_a = comparator.detect_faces(image_a)
    faces_b = comparator.detect_faces(image_b)

    sim_matrix = comparator.compare_sets(faces_a, faces_b)

    print("Shape matriz:", sim_matrix.shape)
    print(sim_matrix)

    plot_similarity_heatmap(
        sim_matrix=sim_matrix,
        faces_a=faces_a,
        faces_b=faces_b,
        output_path=output_plot
    )