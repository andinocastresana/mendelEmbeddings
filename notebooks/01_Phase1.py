# %%
# =========================================
# ID: FASE1_001
# VERSION: v1.1
# =========================================
# Descripción:
# - Importa librerías básicas
# - Muestra el directorio de trabajo actual
# - Define una carga de imágenes más robusta usando pathlib
# - Verifica explícitamente si los archivos existen antes de abrirlos

import os
import cv2
import numpy as np
from pathlib import Path


def load_image(image_path: str | Path) -> np.ndarray:
    """
    Carga una imagen desde disco y la convierte a RGB.

    Args:
        image_path: ruta absoluta o relativa a la imagen

    Returns:
        np.ndarray: imagen en formato RGB
    """
    image_path = Path(image_path)

    # Resolver la ruta para ver exactamente qué archivo se intenta abrir
    image_path_resolved = image_path.resolve()

    if not image_path.exists():
        raise FileNotFoundError(
            f"No existe el archivo:\n"
            f"  ruta dada: {image_path}\n"
            f"  ruta resuelta: {image_path_resolved}\n"
            f"  cwd actual: {Path.cwd()}"
        )

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(
            f"El archivo existe pero OpenCV no pudo leerlo:\n"
            f"  ruta: {image_path_resolved}"
        )

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


# =========================
# Diagnóstico de rutas
# =========================
print("Directorio de trabajo actual (cwd):")
print(Path.cwd())
print()

# Ajusta estas rutas según tu proyecto
img1_path = Path("../data/input/img/fraternos_jovenes.jpg")
img2_path = Path("../data/input/img/fraternosChacabuco8.jpg")

print("Chequeo de existencia:")
print(f"{img1_path} -> {img1_path.exists()} | resuelta: {img1_path.resolve()}")
print(f"{img2_path} -> {img2_path.exists()} | resuelta: {img2_path.resolve()}")
print()

# Cargar imágenes
img1 = load_image(img1_path)
img2 = load_image(img2_path)

print(f"Imagen 1 shape: {img1.shape}")
print(f"Imagen 2 shape: {img2.shape}")

# %%
# =========================================
# ID: FASE1_002
# VERSION: v1.0
# =========================================
# Descripción:
# - Detecta todos los rostros en varias imágenes
# - Asigna un ID a cada rostro según la foto de origen
# - Dibuja las cajas sobre cada imagen
# - Guarda metadatos y recortes para poder elegir luego qué pares comparar
#
# Requisitos:
# pip install insightface onnxruntime matplotlib
#
# Supone que ya existen en memoria:
# - img1
# - img2
# cargadas como np.ndarray en RGB (del bloque anterior)

import numpy as np
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis

# -------------------------
# 1) Inicializar detector
# -------------------------
# providers=["CPUExecutionProvider"] evita problemas si no usas GPU
app = FaceAnalysis(providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

# -------------------------
# 2) Función de detección
# -------------------------
def detect_faces_in_image(img_rgb: np.ndarray, photo_label: str):
    """
    Detecta todos los rostros de una imagen y devuelve:
    - image_annotated: imagen con cajas e IDs
    - face_records: lista de dicts con metadatos por rostro
    """
    faces = app.get(img_rgb)

    # Copia para dibujar
    image_annotated = img_rgb.copy()
    face_records = []

    # Ordenar de izquierda a derecha para tener IDs reproducibles
    faces_sorted = sorted(faces, key=lambda f: f.bbox[0])

    for i, face in enumerate(faces_sorted, start=1):
        x1, y1, x2, y2 = face.bbox.astype(int)
        face_id = f"{photo_label}_R{i}"

        # Recorte del rostro
        crop = img_rgb[y1:y2, x1:x2].copy()

        # Guardar metadatos útiles para etapas posteriores
        record = {
            "face_id": face_id,
            "photo_label": photo_label,
            "bbox": (x1, y1, x2, y2),
            "det_score": float(face.det_score),
            "crop_rgb": crop,
            "embedding": face.embedding.copy(),   # útil para comparación global posterior
            "kps": face.kps.copy() if hasattr(face, "kps") else None,  # landmarks básicos
        }
        face_records.append(record)

        # Dibujar caja
        image_annotated[y1:y1+2, x1:x2] = [255, 0, 0]
        image_annotated[y2-2:y2, x1:x2] = [255, 0, 0]
        image_annotated[y1:y2, x1:x1+2] = [255, 0, 0]
        image_annotated[y1:y2, x2-2:x2] = [255, 0, 0]

        # Escribir ID con matplotlib luego; aquí solo dejamos el registro
        print(f"{face_id} | foto={photo_label} | bbox={(x1, y1, x2, y2)} | score={face.det_score:.3f}")

    return image_annotated, face_records


# -------------------------
# 3) Detectar en ambas fotos
# -------------------------
images = {
    "F1": img1,
    "F2": img2,
}

all_face_records = []
annotated_images = {}

for photo_label, img_rgb in images.items():
    annotated_img, records = detect_faces_in_image(img_rgb, photo_label)
    annotated_images[photo_label] = annotated_img
    all_face_records.extend(records)

# -------------------------
# 4) Visualizar resultados
# -------------------------
fig, axes = plt.subplots(1, len(images), figsize=(8 * len(images), 8))

# Asegurar iterabilidad si solo hubiera una imagen
if len(images) == 1:
    axes = [axes]

for ax, (photo_label, img_rgb) in zip(axes, annotated_images.items()):
    ax.imshow(img_rgb)
    ax.set_title(f"{photo_label} - rostros detectados")
    ax.axis("off")

    # Añadir etiquetas de ID sobre cada bbox
    for rec in all_face_records:
        if rec["photo_label"] == photo_label:
            x1, y1, x2, y2 = rec["bbox"]
            ax.text(
                x1, max(10, y1 - 10), rec["face_id"],
                fontsize=12,
                bbox=dict(facecolor="yellow", alpha=0.7, edgecolor="black")
            )

plt.tight_layout()
plt.show()

# -------------------------
# 5) Resumen seleccionable
# -------------------------
print("\nResumen de rostros detectados:")
for rec in all_face_records:
    print(f"- {rec['face_id']}  <-  {rec['photo_label']}")

# Ejemplo:
# Luego podrás elegir algo como:
# face_a = next(r for r in all_face_records if r["face_id"] == "F1_R1")
# face_b = next(r for r in all_face_records if r["face_id"] == "F2_R2")

# %%
# =========================================
# ID: FASE1_003
# VERSION: v1.0
# =========================================
# Descripción:
# - Selecciona un par de rostros detectados previamente
# - Alinea ambos usando los 5 keypoints de InsightFace
# - Visualiza:
#     1) recorte original
#     2) recorte con keypoints
#     3) rostro alineado
#
# Requisitos:
# - Haber ejecutado el bloque FASE1_002
# - Debe existir `all_face_records`
#
# Nota:
# - InsightFace expone utilidades de alineación basadas en landmarks
#   mediante face_align.norm_crop()

import matplotlib.pyplot as plt
import numpy as np
from insightface.utils import face_align

# -------------------------
# 1) Elegir el par
# -------------------------
# Cambia estos IDs según los rostros que quieras comparar
face_id_a = "F1_R1"
face_id_b = "F2_R1"

# Buscar los registros correspondientes
face_a = next(r for r in all_face_records if r["face_id"] == face_id_a)
face_b = next(r for r in all_face_records if r["face_id"] == face_id_b)

# -------------------------
# 2) Funciones auxiliares
# -------------------------
def align_face_from_record(face_record: dict, image_size: int = 112) -> np.ndarray:
    """
    Alinea un rostro usando los keypoints detectados por InsightFace.

    Args:
        face_record (dict): registro de rostro generado en FASE1_002
        image_size (int): tamaño del rostro alineado de salida

    Returns:
        np.ndarray: rostro alineado en RGB
    """
    # En FASE1_002 guardamos el recorte y los keypoints locales al rostro detectado.
    crop_rgb = face_record["crop_rgb"]
    kps = face_record["kps"]

    if kps is None:
        raise ValueError(f"El rostro {face_record['face_id']} no tiene keypoints disponibles.")

    # Ajustar keypoints al sistema de coordenadas del recorte
    # bbox = (x1, y1, x2, y2) en coordenadas de la imagen completa
    x1, y1, _, _ = face_record["bbox"]
    kps_local = kps.copy()
    kps_local[:, 0] -= x1
    kps_local[:, 1] -= y1

    # Alinear rostro
    aligned_rgb = face_align.norm_crop(crop_rgb, landmark=kps_local, image_size=image_size)
    return aligned_rgb


def plot_face_triplet(face_record: dict, aligned_rgb: np.ndarray):
    """
    Muestra:
    1) recorte original
    2) recorte con keypoints
    3) rostro alineado
    """
    crop_rgb = face_record["crop_rgb"]
    kps = face_record["kps"]

    x1, y1, _, _ = face_record["bbox"]
    kps_local = kps.copy()
    kps_local[:, 0] -= x1
    kps_local[:, 1] -= y1

    # Crear copia para visualizar landmarks
    crop_with_kps = crop_rgb.copy()

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    # Panel 1: recorte original
    axes[0].imshow(crop_rgb)
    axes[0].set_title(f"{face_record['face_id']}\nrecorte")
    axes[0].axis("off")

    # Panel 2: recorte + keypoints
    axes[1].imshow(crop_with_kps)
    axes[1].scatter(
        kps_local[:, 0],
        kps_local[:, 1],
        s=40
    )
    axes[1].set_title(f"{face_record['face_id']}\nkeypoints")
    axes[1].axis("off")

    # Panel 3: alineado
    axes[2].imshow(aligned_rgb)
    axes[2].set_title(f"{face_record['face_id']}\nalineado")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


# -------------------------
# 3) Alinear ambos rostros
# -------------------------
aligned_a = align_face_from_record(face_a, image_size=112)
aligned_b = align_face_from_record(face_b, image_size=112)

# -------------------------
# 4) Visualizar alineación
# -------------------------
plot_face_triplet(face_a, aligned_a)
plot_face_triplet(face_b, aligned_b)

# -------------------------
# 5) Guardar resultados para el siguiente bloque
# -------------------------
selected_pair = {
    "face_a": face_a,
    "face_b": face_b,
    "aligned_a": aligned_a,
    "aligned_b": aligned_b,
}

print(f"Par seleccionado: {face_id_a} vs {face_id_b}")
print(f"Shapes alineados: {aligned_a.shape} vs {aligned_b.shape}")

# %%
# =========================================
# ID: FASE1_004
# VERSION: v1.0
# =========================================
# Descripción:
# - Toma el par ya alineado del bloque anterior
# - Recalcula el embedding global sobre los rostros alineados
# - Conserva y compara contra los embeddings guardados en detección
# - Visualiza una parte de ambos embeddings para inspección rápida
#
# Requisitos:
# - Haber ejecutado FASE1_002 y FASE1_003
# - Deben existir:
#     app
#     selected_pair
#
# Nota:
# - El embedding "original" aquí es el que guardamos durante la detección
# - El embedding "post_align" es el recalculado explícitamente sobre aligned_a/aligned_b

import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 1) Recuperar el par
# -------------------------
face_a = selected_pair["face_a"]
face_b = selected_pair["face_b"]
aligned_a = selected_pair["aligned_a"]
aligned_b = selected_pair["aligned_b"]

# -------------------------
# 2) Localizar el modelo de reconocimiento dentro de FaceAnalysis
# -------------------------
def get_recognition_model(face_app):
    """
    Busca dentro de FaceAnalysis el submodelo que expone get_feat(),
    que es el que genera embeddings faciales.
    """
    for _, model in face_app.models.items():
        if hasattr(model, "get_feat"):
            return model
    raise RuntimeError("No se encontró un modelo de reconocimiento con método get_feat().")

rec_model = get_recognition_model(app)

# -------------------------
# 3) Funciones auxiliares
# -------------------------
def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """
    Normaliza un vector a norma L2 = 1.
    """
    vec = np.asarray(vec).astype(np.float32).ravel()
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calcula similitud coseno entre dos vectores.
    """
    v1 = l2_normalize(vec1)
    v2 = l2_normalize(vec2)
    return float(np.dot(v1, v2))


def extract_embedding_from_aligned(aligned_rgb: np.ndarray) -> np.ndarray:
    """
    Extrae embedding desde un rostro ya alineado.
    """
    emb = rec_model.get_feat(aligned_rgb).flatten()
    return emb.astype(np.float32)


# -------------------------
# 4) Embeddings originales (detección) y post-alineación
# -------------------------
emb_a_original = np.asarray(face_a["embedding"]).astype(np.float32).ravel()
emb_b_original = np.asarray(face_b["embedding"]).astype(np.float32).ravel()

emb_a_post_align = extract_embedding_from_aligned(aligned_a)
emb_b_post_align = extract_embedding_from_aligned(aligned_b)

# Normalizados (más cómodos para comparar)
emb_a_original_norm = l2_normalize(emb_a_original)
emb_b_original_norm = l2_normalize(emb_b_original)
emb_a_post_align_norm = l2_normalize(emb_a_post_align)
emb_b_post_align_norm = l2_normalize(emb_b_post_align)

# -------------------------
# 5) Comparaciones útiles
# -------------------------
# A) Estabilidad del embedding de cada rostro:
#    original vs post-align
self_sim_a = cosine_similarity(emb_a_original, emb_a_post_align)
self_sim_b = cosine_similarity(emb_b_original, emb_b_post_align)

# B) Similitud entre ambos rostros:
#    usando embedding original y usando embedding post-align
pair_sim_original = cosine_similarity(emb_a_original, emb_b_original)
pair_sim_post_align = cosine_similarity(emb_a_post_align, emb_b_post_align)

# Cambio absoluto en la similitud del par tras recalcular embedding
pair_similarity_delta = pair_sim_post_align - pair_sim_original

# -------------------------
# 6) Guardar resultados para bloques siguientes
# -------------------------
selected_pair["embedding_a_original"] = emb_a_original
selected_pair["embedding_b_original"] = emb_b_original
selected_pair["embedding_a_post_align"] = emb_a_post_align
selected_pair["embedding_b_post_align"] = emb_b_post_align

selected_pair["embedding_qc"] = {
    "self_similarity_a_original_vs_post_align": self_sim_a,
    "self_similarity_b_original_vs_post_align": self_sim_b,
    "pair_similarity_original": pair_sim_original,
    "pair_similarity_post_align": pair_sim_post_align,
    "pair_similarity_delta": pair_similarity_delta,
}

# -------------------------
# 7) Resumen numérico
# -------------------------
print("=== CONTROL DE CALIDAD DEL EMBEDDING ===")
print(f"{face_a['face_id']} | similitud original vs post_align: {self_sim_a:.4f}")
print(f"{face_b['face_id']} | similitud original vs post_align: {self_sim_b:.4f}")
print()
print("=== SIMILITUD DEL PAR ===")
print(f"Par {face_a['face_id']} vs {face_b['face_id']} | embedding original:   {pair_sim_original:.4f}")
print(f"Par {face_a['face_id']} vs {face_b['face_id']} | embedding post_align: {pair_sim_post_align:.4f}")
print(f"Diferencia (post_align - original): {pair_similarity_delta:+.4f}")

# Interpretación práctica rápida:
# - self_similarity cercana a 1.0 => la alineación no alteró mucho el embedding
# - self_similarity más baja => posible sensibilidad a pose / recorte / calidad
# - pair_similarity_post_align será la que conviene usar como referencia principal

# -------------------------
# 8) Visualización rápida del embedding
# -------------------------
# Mostramos solo las primeras dimensiones para inspección visual
n_dims_to_plot = min(64, len(emb_a_original_norm))
x = np.arange(n_dims_to_plot)

fig, axes = plt.subplots(2, 2, figsize=(12, 7))

# Rostro A
axes[0, 0].plot(x, emb_a_original_norm[:n_dims_to_plot], label="original")
axes[0, 0].plot(x, emb_a_post_align_norm[:n_dims_to_plot], label="post_align")
axes[0, 0].set_title(f"{face_a['face_id']} - primeras {n_dims_to_plot} dims")
axes[0, 0].legend()

# Rostro B
axes[0, 1].plot(x, emb_b_original_norm[:n_dims_to_plot], label="original")
axes[0, 1].plot(x, emb_b_post_align_norm[:n_dims_to_plot], label="post_align")
axes[0, 1].set_title(f"{face_b['face_id']} - primeras {n_dims_to_plot} dims")
axes[0, 1].legend()

# Diferencia absoluta por dimensión (A)
axes[1, 0].plot(x, np.abs(emb_a_original_norm[:n_dims_to_plot] - emb_a_post_align_norm[:n_dims_to_plot]))
axes[1, 0].set_title(f"{face_a['face_id']} - |original - post_align|")

# Diferencia absoluta por dimensión (B)
axes[1, 1].plot(x, np.abs(emb_b_original_norm[:n_dims_to_plot] - emb_b_post_align_norm[:n_dims_to_plot]))
axes[1, 1].set_title(f"{face_b['face_id']} - |original - post_align|")

plt.tight_layout()
plt.show()

# %%
# =========================================
# ID: FASE1_005
# VERSION: v1.0
# =========================================
# Descripción:
# - Calcula el score global entre los dos rostros seleccionados
# - Usa como referencia principal los embeddings post-alineación
# - Calcula también métricas de control con los embeddings originales
# - Muestra un resumen simple y una visualización compacta
#
# Requisitos:
# - Haber ejecutado FASE1_004
# - Debe existir `selected_pair`

import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 1) Recuperar embeddings
# -------------------------
emb_a_original = selected_pair["embedding_a_original"]
emb_b_original = selected_pair["embedding_b_original"]
emb_a_post = selected_pair["embedding_a_post_align"]
emb_b_post = selected_pair["embedding_b_post_align"]

face_a = selected_pair["face_a"]
face_b = selected_pair["face_b"]

# -------------------------
# 2) Funciones de distancia
# -------------------------
def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """
    Normaliza un vector a norma L2 = 1.
    """
    vec = np.asarray(vec, dtype=np.float32).ravel()
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Similitud coseno entre dos vectores.
    """
    v1 = l2_normalize(vec1)
    v2 = l2_normalize(vec2)
    return float(np.dot(v1, v2))


def cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Distancia coseno = 1 - similitud coseno.
    """
    return 1.0 - cosine_similarity(vec1, vec2)


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Distancia euclídea entre embeddings normalizados.
    """
    v1 = l2_normalize(vec1)
    v2 = l2_normalize(vec2)
    return float(np.linalg.norm(v1 - v2))


# -------------------------
# 3) Calcular scores
# -------------------------
scores = {
    "cosine_similarity_original": cosine_similarity(emb_a_original, emb_b_original),
    "cosine_similarity_post_align": cosine_similarity(emb_a_post, emb_b_post),
    "cosine_distance_original": cosine_distance(emb_a_original, emb_b_original),
    "cosine_distance_post_align": cosine_distance(emb_a_post, emb_b_post),
    "euclidean_distance_original": euclidean_distance(emb_a_original, emb_b_original),
    "euclidean_distance_post_align": euclidean_distance(emb_a_post, emb_b_post),
}

# Guardar para bloques posteriores
selected_pair["global_scores"] = scores

# -------------------------
# 4) Resumen numérico
# -------------------------
print("=== COMPARACIÓN GLOBAL DEL PAR ===")
print(f"Rostro A: {face_a['face_id']}")
print(f"Rostro B: {face_b['face_id']}")
print()
print(f"Cosine similarity (original):   {scores['cosine_similarity_original']:.4f}")
print(f"Cosine similarity (post_align): {scores['cosine_similarity_post_align']:.4f}")
print()
print(f"Cosine distance   (original):   {scores['cosine_distance_original']:.4f}")
print(f"Cosine distance   (post_align): {scores['cosine_distance_post_align']:.4f}")
print()
print(f"Euclidean distance (original):   {scores['euclidean_distance_original']:.4f}")
print(f"Euclidean distance (post_align): {scores['euclidean_distance_post_align']:.4f}")

# Nota práctica:
# - mayor cosine similarity  => mayor parecido
# - menor cosine distance    => mayor parecido
# - menor euclidean distance => mayor parecido

# -------------------------
# 5) Visualización compacta
# -------------------------
metric_names = [
    "cosine_similarity_original",
    "cosine_similarity_post_align",
    "cosine_distance_original",
    "cosine_distance_post_align",
    "euclidean_distance_original",
    "euclidean_distance_post_align",
]
metric_values = [scores[k] for k in metric_names]

plt.figure(figsize=(10, 4))
plt.bar(range(len(metric_names)), metric_values)
plt.xticks(range(len(metric_names)), metric_names, rotation=45, ha="right")
plt.title(f"Scores globales: {face_a['face_id']} vs {face_b['face_id']}")
plt.tight_layout()
plt.show()

# %%
#Prueba rápida
import cv2
import numpy as np
import mediapipe as mp
from insightface.app import FaceAnalysis

print("cv2 version:", cv2.__version__)
print("numpy version:", np.__version__)
print("mediapipe version:", mp.__version__)

# Prueba básica con una imagen ya cargada
print("img1 shape:", img1.shape)
print("img2 shape:", img2.shape)

# Prueba básica de InsightFace
app_test = FaceAnalysis(providers=["CPUExecutionProvider"])
app_test.prepare(ctx_id=0, det_size=(640, 640))
faces_test = app_test.get(img1)

print("Rostros detectados en img1:", len(faces_test))

# %%
# =========================================
# ID: FASE1_006
# VERSION: v1.0
# =========================================
# Descripción:
# - Detecta landmarks faciales densos (MediaPipe) sobre los rostros alineados
# - Define regiones básicas del rostro
# - Visualiza landmarks y regiones para validación
#
# Requisitos:
# pip install mediapipe
# - Haber ejecutado FASE1_003
# - Debe existir selected_pair con aligned_a y aligned_b

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

# -------------------------
# 1) Inicializar MediaPipe
# -------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    refine_landmarks=True,
    max_num_faces=1
)

# -------------------------
# 2) Obtener landmarks
# -------------------------
def get_landmarks(img_rgb):
    """
    Devuelve landmarks como array Nx2 en coordenadas de imagen
    """
    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        raise ValueError("No se detectaron landmarks.")

    h, w, _ = img_rgb.shape
    landmarks = []

    for lm in results.multi_face_landmarks[0].landmark:
        x = int(lm.x * w)
        y = int(lm.y * h)
        landmarks.append([x, y])

    return np.array(landmarks)


# -------------------------
# 3) Definir regiones simples (índices MediaPipe)
# -------------------------
REGIONS = {
    "left_eye": [33, 133, 160, 159, 158, 157],
    "right_eye": [362, 263, 387, 386, 385, 384],
    "nose": [1, 2, 98, 327],
    "mouth": [61, 291, 0, 17],
    "forehead": [10, 151, 9],
    "jaw": [152, 234, 454]
}


# -------------------------
# 4) Visualización
# -------------------------
def plot_landmarks_and_regions(img_rgb, landmarks, title):
    img = img_rgb.copy()

    plt.figure(figsize=(5,5))
    plt.imshow(img)

    # landmarks
    plt.scatter(landmarks[:,0], landmarks[:,1], s=5)

    # regiones
    for name, idxs in REGIONS.items():
        pts = landmarks[idxs]
        plt.scatter(pts[:,0], pts[:,1], s=30, label=name)

    plt.title(title)
    plt.legend()
    plt.axis("off")
    plt.show()


# -------------------------
# 5) Aplicar a ambos rostros
# -------------------------
aligned_a = selected_pair["aligned_a"]
aligned_b = selected_pair["aligned_b"]

landmarks_a = get_landmarks(aligned_a)
landmarks_b = get_landmarks(aligned_b)

# guardar para siguientes bloques
selected_pair["landmarks_a"] = landmarks_a
selected_pair["landmarks_b"] = landmarks_b

# visualizar
plot_landmarks_and_regions(aligned_a, landmarks_a, "Face A - landmarks")
plot_landmarks_and_regions(aligned_b, landmarks_b, "Face B - landmarks")

# %%
# =========================================
# ID: FASE1_007
# VERSION: v1.1
# =========================================
# Mejora:
# - Regiones mucho más precisas usando subsets reales de FaceMesh
# - Evita polígonos gigantes
# - Mejor separación anatómica

import numpy as np
import cv2
import matplotlib.pyplot as plt
'''
# -------------------------
# 1) REGIONES MEJOR DEFINIDAS
# -------------------------
REGIONS = {
    # contorno ojos (más cerrado)
    "left_eye": [33, 160, 158, 133, 153, 144],
    "right_eye": [362, 385, 387, 263, 373, 380],

    # nariz (puente + base)
    "nose": [1, 2, 98, 327, 168, 197],

    # boca (labios)
    "mouth": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],

    # frente (más localizada)
    "forehead": [10, 151, 9, 8],

    # mandíbula inferior
    "jaw": [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 454]
}
'''
# VERSION: v1.2
# =========================================
# Mejora:
# - Elimina jaw (demasiado ruidoso)
# - Mejora forehead
# - Mantiene regiones limpias y útiles

REGIONS = {
    "left_eye": [33, 160, 158, 133, 153, 144],
    "right_eye": [362, 385, 387, 263, 373, 380],

    "nose": [1, 2, 98, 327, 168, 197],

    "mouth": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],

    # frente mejor definida
    "forehead": [10, 109, 67, 103, 151, 9, 8]
}



# -------------------------
# 2) Máscara robusta
# -------------------------
def create_region_mask(img_shape, landmarks, indices):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)

    pts = landmarks[indices].astype(np.int32)

    # evitar regiones degeneradas
    if len(pts) < 3:
        return mask

    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)

    return mask


def generate_region_masks(img, landmarks):
    masks = {}
    for name, idxs in REGIONS.items():
        masks[name] = create_region_mask(img.shape, landmarks, idxs)
    return masks


# -------------------------
# 3) Overlay mejorado
# -------------------------
def overlay_regions(img, masks):
    overlay = img.copy()

    colors = {
        "left_eye": (255, 0, 0),
        "right_eye": (0, 255, 0),
        "nose": (0, 0, 255),
        "mouth": (255, 255, 0),
        "forehead": (255, 0, 255),
        "jaw": (0, 255, 255)
    }

    for name, mask in masks.items():
        color = np.array(colors.get(name, (200, 200, 200)))

        # suavizar impacto visual
        overlay[mask > 0] = (
            0.7 * overlay[mask > 0] + 0.3 * color
        ).astype(np.uint8)

    return overlay


# -------------------------
# 4) Aplicar
# -------------------------
aligned_a = selected_pair["aligned_a"]
aligned_b = selected_pair["aligned_b"]

landmarks_a = selected_pair["landmarks_a"]
landmarks_b = selected_pair["landmarks_b"]

masks_a = generate_region_masks(aligned_a, landmarks_a)
masks_b = generate_region_masks(aligned_b, landmarks_b)

selected_pair["masks_a"] = masks_a
selected_pair["masks_b"] = masks_b

# -------------------------
# 5) Visualizar
# -------------------------
overlay_a = overlay_regions(aligned_a, masks_a)
overlay_b = overlay_regions(aligned_b, masks_b)

fig, axes = plt.subplots(1, 2, figsize=(10,5))

axes[0].imshow(overlay_a)
axes[0].set_title("Face A - regiones refinadas")

axes[1].imshow(overlay_b)
axes[1].set_title("Face B - regiones refinadas")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()

# %%
'''
# =========================================
# ID: FASE1_008
# VERSION: v1.0
# =========================================
# Descripción:
# - Extrae parches por región a partir de las máscaras
# - Calcula similitud regional entre ambos rostros
# - Genera un "heatmap" simple sobre cada rostro
#
# Requisitos:
# - Haber ejecutado FASE1_007
# - Deben existir en selected_pair:
#     aligned_a, aligned_b, masks_a, masks_b
#
# Nota:
# - Aquí usamos una métrica visual simple y robusta para empezar:
#   comparación por intensidad sobre parches normalizados
# - Más adelante podremos reemplazarla por embeddings locales

import numpy as np
import cv2
import matplotlib.pyplot as plt

# -------------------------
# 1) Recuperar datos
# -------------------------
aligned_a = selected_pair["aligned_a"]
aligned_b = selected_pair["aligned_b"]
masks_a = selected_pair["masks_a"]
masks_b = selected_pair["masks_b"]

# -------------------------
# 2) Utilidades
# -------------------------
def extract_region_patch(img_rgb, mask, out_size=(64, 64)):
    """
    Extrae el parche delimitado por la máscara y lo redimensiona.
    Devuelve:
    - patch_rgb: parche RGB redimensionado
    - bbox: caja original (x1, y1, x2, y2)
    """
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None, None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    # pequeño margen alrededor de la región
    pad = 4
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img_rgb.shape[1] - 1, x2 + pad)
    y2 = min(img_rgb.shape[0] - 1, y2 + pad)

    patch = img_rgb[y1:y2+1, x1:x2+1].copy()

    if patch.size == 0:
        return None, None

    patch_resized = cv2.resize(patch, out_size, interpolation=cv2.INTER_LINEAR)
    return patch_resized, (x1, y1, x2, y2)


def patch_similarity(patch_a_rgb, patch_b_rgb):
    """
    Similitud simple entre parches.
    Estrategia:
    - pasar a escala de grises
    - normalizar intensidades
    - calcular similitud coseno
    """
    gray_a = cv2.cvtColor(patch_a_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32).ravel()
    gray_b = cv2.cvtColor(patch_b_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32).ravel()

    # normalización z-score para reducir efecto de brillo global
    gray_a = (gray_a - gray_a.mean()) / (gray_a.std() + 1e-8)
    gray_b = (gray_b - gray_b.mean()) / (gray_b.std() + 1e-8)

    denom = (np.linalg.norm(gray_a) * np.linalg.norm(gray_b)) + 1e-8
    sim = float(np.dot(gray_a, gray_b) / denom)

    # acotar por seguridad
    sim = max(-1.0, min(1.0, sim))
    return sim


def similarity_to_color(sim):
    """
    Convierte similitud [-1, 1] a color RGB:
    - rojo = baja similitud
    - amarillo = intermedia
    - verde = alta similitud
    """
    t = (sim + 1.0) / 2.0  # a [0,1]

    if t < 0.5:
        # rojo -> amarillo
        alpha = t / 0.5
        color = np.array([255, int(255 * alpha), 0], dtype=np.uint8)
    else:
        # amarillo -> verde
        alpha = (t - 0.5) / 0.5
        color = np.array([int(255 * (1 - alpha)), 255, 0], dtype=np.uint8)

    return color


def overlay_similarity_map(img_rgb, masks_dict, region_scores, alpha=0.35):
    """
    Superpone color por región en función del score de similitud.
    """
    overlay = img_rgb.copy()

    for region_name, mask in masks_dict.items():
        if region_name not in region_scores:
            continue

        color = similarity_to_color(region_scores[region_name]).astype(np.float32)
        idx = mask > 0
        overlay[idx] = ((1 - alpha) * overlay[idx] + alpha * color).astype(np.uint8)

    return overlay

# -------------------------
# 3) Extraer y comparar regiones
# -------------------------
region_scores = {}
region_patches = {}

common_regions = sorted(set(masks_a.keys()).intersection(set(masks_b.keys())))

for region_name in common_regions:
    patch_a, bbox_a = extract_region_patch(aligned_a, masks_a[region_name], out_size=(64, 64))
    patch_b, bbox_b = extract_region_patch(aligned_b, masks_b[region_name], out_size=(64, 64))

    if patch_a is None or patch_b is None:
        continue

    sim = patch_similarity(patch_a, patch_b)

    region_scores[region_name] = sim
    region_patches[region_name] = {
        "patch_a": patch_a,
        "patch_b": patch_b,
        "bbox_a": bbox_a,
        "bbox_b": bbox_b,
    }

# Guardar para bloques posteriores
selected_pair["region_scores"] = region_scores
selected_pair["region_patches"] = region_patches

# -------------------------
# 4) Mostrar scores
# -------------------------
print("=== SIMILITUD POR REGIÓN ===")
for region_name, score in sorted(region_scores.items()):
    print(f"{region_name:10s}: {score:.4f}")

# -------------------------
# 5) Generar heatmap sobre ambos rostros
# -------------------------
heatmap_a = overlay_similarity_map(aligned_a, masks_a, region_scores, alpha=0.35)
heatmap_b = overlay_similarity_map(aligned_b, masks_b, region_scores, alpha=0.35)

# -------------------------
# 6) Visualización principal
# -------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(heatmap_a)
axes[0].set_title("Face A - similitud regional")
axes[0].axis("off")

axes[1].imshow(heatmap_b)
axes[1].set_title("Face B - similitud regional")
axes[1].axis("off")

plt.tight_layout()
plt.show()

# -------------------------
# 7) Visualización de parches
# -------------------------
n_regions = len(region_patches)

if n_regions > 0:
    fig, axes = plt.subplots(n_regions, 2, figsize=(6, 3 * n_regions))

    # Si hay una sola región, axes no viene como matriz
    if n_regions == 1:
        axes = np.array([axes])

    for row_idx, region_name in enumerate(sorted(region_patches.keys())):
        patch_a = region_patches[region_name]["patch_a"]
        patch_b = region_patches[region_name]["patch_b"]
        score = region_scores[region_name]

        axes[row_idx, 0].imshow(patch_a)
        axes[row_idx, 0].set_title(f"{region_name} - A")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(patch_b)
        axes[row_idx, 1].set_title(f"{region_name} - B | sim={score:.3f}")
        axes[row_idx, 1].axis("off")

    plt.tight_layout()
    plt.show()
    '''

# %%
'''
# =========================================
# ID: FASE1_009
# VERSION: v1.0
# =========================================
# Descripción:
# - Calcula un mapa de occlusion sensitivity sobre Face A
# - Mantiene Face B fija como referencia
# - Mide cuánto cambia la similitud global al ocluir pequeñas ventanas de Face A
# - Superpone el mapa sobre el rostro
#
# Requisitos:
# - Haber ejecutado FASE1_004
# - Deben existir:
#     selected_pair["aligned_a"], selected_pair["aligned_b"]
#     app  (FaceAnalysis ya inicializado)
#
# Nota:
# - Mapa positivo (más intenso) = zonas cuya oclusión baja más la similitud
# - Esas zonas son las que más contribuyen al match global

import numpy as np
import cv2
import matplotlib.pyplot as plt

# -------------------------
# 1) Recuperar datos
# -------------------------
aligned_a = selected_pair["aligned_a"].copy()
aligned_b = selected_pair["aligned_b"].copy()

# -------------------------
# 2) Localizar modelo de embeddings
# -------------------------
def get_recognition_model(face_app):
    """
    Busca dentro de FaceAnalysis el modelo que expone get_feat().
    """
    for _, model in face_app.models.items():
        if hasattr(model, "get_feat"):
            return model
    raise RuntimeError("No se encontró un modelo de reconocimiento con método get_feat().")

rec_model = get_recognition_model(app)

# -------------------------
# 3) Utilidades
# -------------------------
def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """
    Normaliza un vector a norma L2 = 1.
    """
    vec = np.asarray(vec, dtype=np.float32).ravel()
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Similitud coseno entre dos embeddings.
    """
    v1 = l2_normalize(vec1)
    v2 = l2_normalize(vec2)
    return float(np.dot(v1, v2))


def extract_embedding_from_aligned(aligned_rgb: np.ndarray) -> np.ndarray:
    """
    Extrae embedding desde un rostro ya alineado.
    """
    emb = rec_model.get_feat(aligned_rgb).flatten()
    return emb.astype(np.float32)


def occlude_patch(img_rgb: np.ndarray, x: int, y: int, w: int, h: int, fill_value: int = 127) -> np.ndarray:
    """
    Devuelve una copia de la imagen con una ventana ocluida.
    """
    out = img_rgb.copy()
    out[y:y+h, x:x+w] = fill_value
    return out


def make_heat_overlay(img_rgb: np.ndarray, heatmap_01: np.ndarray, alpha: float = 0.40) -> np.ndarray:
    """
    Superpone un heatmap sobre la imagen RGB.
    """
    heat_uint8 = np.uint8(np.clip(heatmap_01, 0, 1) * 255)
    heat_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    overlay = ((1 - alpha) * img_rgb + alpha * heat_rgb).astype(np.uint8)
    return overlay

# -------------------------
# 4) Embedding y score base
# -------------------------
emb_a_base = extract_embedding_from_aligned(aligned_a)
emb_b_base = extract_embedding_from_aligned(aligned_b)
base_similarity = cosine_similarity(emb_a_base, emb_b_base)

print(f"Base cosine similarity (A vs B): {base_similarity:.4f}")

# -------------------------
# 5) Parámetros de oclusión
# -------------------------
win_size = 16   # tamaño de ventana
stride = 4      # paso
fill_value = 127  # gris medio

h, w, _ = aligned_a.shape

# Acumuladores para promediar contribuciones por píxel
delta_sum = np.zeros((h, w), dtype=np.float32)
delta_count = np.zeros((h, w), dtype=np.float32)

# -------------------------
# 6) Barrido de oclusión
# -------------------------
for y in range(0, h - win_size + 1, stride):
    for x in range(0, w - win_size + 1, stride):
        occluded_a = occlude_patch(aligned_a, x, y, win_size, win_size, fill_value=fill_value)

        emb_occ = extract_embedding_from_aligned(occluded_a)
        sim_occ = cosine_similarity(emb_occ, emb_b_base)

        # Cuánto cae la similitud al tapar esta zona
        delta = base_similarity - sim_occ

        # Acumular sobre la ventana afectada
        delta_sum[y:y+win_size, x:x+win_size] += delta
        delta_count[y:y+win_size, x:x+win_size] += 1.0

# Evitar divisiones por cero
delta_mean = delta_sum / np.maximum(delta_count, 1e-8)

# -------------------------
# 7) Normalización del mapa
# -------------------------
# Nos interesan sobre todo las caídas positivas de similitud
delta_pos = np.maximum(delta_mean, 0)

if delta_pos.max() > 0:
    heatmap_01 = delta_pos / delta_pos.max()
else:
    heatmap_01 = delta_pos.copy()

# Guardar resultados
selected_pair["occlusion_A_vs_B"] = {
    "base_similarity": base_similarity,
    "win_size": win_size,
    "stride": stride,
    "fill_value": fill_value,
    "delta_mean": delta_mean,
    "heatmap_01": heatmap_01,
}

# -------------------------
# 8) Overlay visual
# -------------------------
overlay = make_heat_overlay(aligned_a, heatmap_01, alpha=0.40)

# -------------------------
# 9) Visualización
# -------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].imshow(aligned_a)
axes[0].set_title("Face A")
axes[0].axis("off")

axes[1].imshow(heatmap_01, cmap="jet")
axes[1].set_title("Occlusion map")
axes[1].axis("off")

axes[2].imshow(overlay)
axes[2].set_title("Face A + heatmap")
axes[2].axis("off")

plt.tight_layout()
plt.show()
'''

# %%
# =========================================
# ID: FASE1_009
# VERSION: v1.3
# =========================================
# Descripción:
# - Reconstruye el signed heatmap si no estaba guardado
# - Genera la superposición sobre la imagen original alineada
# - Permite controlar la transparencia con `alpha`
#
# Requisitos:
# - Haber ejecutado FASE1_009 v1.0 al menos una vez
# - Deben existir:
#     selected_pair["aligned_a"]
#     selected_pair["occlusion_A_vs_B"]["delta_mean"]

import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 1) Recuperar datos
# -------------------------
aligned_a = selected_pair["aligned_a"]

if "occlusion_A_vs_B" not in selected_pair:
    raise KeyError("No existe selected_pair['occlusion_A_vs_B']. Ejecuta antes FASE1_009 v1.0.")

if "heatmap_signed" in selected_pair["occlusion_A_vs_B"]:
    heatmap_signed = selected_pair["occlusion_A_vs_B"]["heatmap_signed"]
elif "delta_mean" in selected_pair["occlusion_A_vs_B"]:
    delta_mean = selected_pair["occlusion_A_vs_B"]["delta_mean"]
    max_abs = np.max(np.abs(delta_mean)) + 1e-8
    heatmap_signed = delta_mean / max_abs
    selected_pair["occlusion_A_vs_B"]["heatmap_signed"] = heatmap_signed
else:
    raise KeyError(
        "No existe ni 'heatmap_signed' ni 'delta_mean' en selected_pair['occlusion_A_vs_B'].\n"
        "Vuelve a ejecutar FASE1_009 v1.0."
    )

# -------------------------
# 2) Conversión signed heatmap -> RGB
# -------------------------
def signed_to_color_map(heatmap: np.ndarray) -> np.ndarray:
    """
    Convierte un heatmap en rango [-1, 1] a RGB:
    -1 -> verde
     0 -> amarillo
     1 -> rojo
    """
    heatmap = np.clip(heatmap, -1.0, 1.0)
    h, w = heatmap.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            v = float(heatmap[y, x])

            if v <= 0:
                # amarillo -> rojo
                t = v
                r = 255
                g = int(255 * (1.0 - t))
                b = 0
            else:
                # verde -> amarillo
                t = -v
                r = int(255 * t)
                g = 255
                b = 0

            out[y, x] = [r, g, b]

    return out

# -------------------------
# 3) Superposición configurable
# -------------------------
def overlay_signed_heatmap(
    base_img_rgb: np.ndarray,
    heatmap_signed: np.ndarray,
    alpha: float = 0.35
) -> np.ndarray:
    """
    Superpone el heatmap firmado sobre la imagen base.

    alpha:
      0.0 = solo imagen
      1.0 = solo heatmap
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha debe estar entre 0.0 y 1.0")

    heat_rgb = signed_to_color_map(heatmap_signed).astype(np.float32)
    base_rgb = base_img_rgb.astype(np.float32)

    overlay = ((1.0 - alpha) * base_rgb + alpha * heat_rgb).astype(np.uint8)
    return overlay

# -------------------------
# 4) Ajustar transparencia
# -------------------------
alpha = 0.35  # cambia aquí

heat_rgb = signed_to_color_map(heatmap_signed)
overlay_signed = overlay_signed_heatmap(aligned_a, heatmap_signed, alpha=alpha)

# Guardar para reutilización
selected_pair["occlusion_A_vs_B"]["overlay_signed"] = overlay_signed
selected_pair["occlusion_A_vs_B"]["overlay_alpha"] = alpha

# -------------------------
# 5) Visualización
# -------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].imshow(aligned_a)
axes[0].set_title("Face A - original")
axes[0].axis("off")

axes[1].imshow(heat_rgb)
axes[1].set_title("Signed heatmap")
axes[1].axis("off")

axes[2].imshow(overlay_signed)
axes[2].set_title(f"Face A + heatmap (alpha={alpha:.2f})")
axes[2].axis("off")

plt.tight_layout()
plt.show()

# %%
# =========================================
# ID: FASE1_010
# VERSION: v1.0
# =========================================
# Descripción:
# - Calcula occlusion sensitivity en ambos sentidos:
#     A ocluida contra B fija
#     B ocluida contra A fija
# - Genera heatmaps firmados:
#     rojo   = esa zona favorece el match
#     verde  = esa zona perjudica el match
#     amarillo = neutra
# - Reorienta el mapa de B al sistema de A y combina ambos
# - Permite controlar la transparencia del overlay con `alpha`
#
# Requisitos:
# - Haber ejecutado FASE1_004
# - Deben existir:
#     selected_pair["aligned_a"], selected_pair["aligned_b"]
#     app

import numpy as np
import cv2
import matplotlib.pyplot as plt

# -------------------------
# 1) Recuperar rostros
# -------------------------
aligned_a = selected_pair["aligned_a"].copy()
aligned_b = selected_pair["aligned_b"].copy()

# -------------------------
# 2) Obtener modelo de embeddings
# -------------------------
def get_recognition_model(face_app):
    """
    Busca dentro de FaceAnalysis el modelo que expone get_feat().
    """
    for _, model in face_app.models.items():
        if hasattr(model, "get_feat"):
            return model
    raise RuntimeError("No se encontró un modelo de reconocimiento con método get_feat().")

rec_model = get_recognition_model(app)

# -------------------------
# 3) Utilidades generales
# -------------------------
def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """
    Normaliza un vector a norma L2 = 1.
    """
    vec = np.asarray(vec, dtype=np.float32).ravel()
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Similitud coseno entre dos embeddings.
    """
    v1 = l2_normalize(vec1)
    v2 = l2_normalize(vec2)
    return float(np.dot(v1, v2))


def extract_embedding_from_aligned(aligned_rgb: np.ndarray) -> np.ndarray:
    """
    Extrae embedding desde un rostro ya alineado.
    """
    emb = rec_model.get_feat(aligned_rgb).flatten()
    return emb.astype(np.float32)


def occlude_patch(img_rgb: np.ndarray, x: int, y: int, w: int, h: int, fill_value: int = 127) -> np.ndarray:
    """
    Devuelve una copia de la imagen con una ventana ocluida.
    """
    out = img_rgb.copy()
    out[y:y+h, x:x+w] = fill_value
    return out


def signed_to_color_map(heatmap: np.ndarray) -> np.ndarray:
    """
    Convierte un heatmap firmado [-1, 1] a RGB:
    -1 -> verde
     0 -> amarillo
     1 -> rojo
    """
    heatmap = np.clip(heatmap, -1.0, 1.0)
    h, w = heatmap.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            v = float(heatmap[y, x])

            if v <= 0:
                # amarillo -> rojo
                t = v
                r = 255
                g = int(255 * (1.0 - t))
                b = 0
            else:
                # verde -> amarillo
                t = -v
                r = int(255 * t)
                g = 255
                b = 0

            out[y, x] = [r, g, b]

    return out


def overlay_signed_heatmap(base_img_rgb: np.ndarray, heatmap_signed: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """
    Superpone el heatmap firmado sobre la imagen base.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha debe estar entre 0.0 y 1.0")

    heat_rgb = signed_to_color_map(heatmap_signed).astype(np.float32)
    base_rgb = base_img_rgb.astype(np.float32)

    overlay = ((1.0 - alpha) * base_rgb + alpha * heat_rgb).astype(np.uint8)
    return overlay


def compute_signed_occlusion_map(
    source_img: np.ndarray,
    reference_img: np.ndarray,
    win_size: int = 16,
    stride: int = 4,
    fill_value: int = 127
):
    """
    Calcula mapa de oclusión firmado para:
    - source_img ocluida
    - reference_img fija

    Signo:
    - delta > 0  -> ocluir empeora el match -> región útil -> rojo
    - delta < 0  -> ocluir mejora el match  -> región perjudicial -> verde
    """
    emb_source = extract_embedding_from_aligned(source_img)
    emb_ref = extract_embedding_from_aligned(reference_img)
    base_similarity = cosine_similarity(emb_source, emb_ref)

    h, w, _ = source_img.shape
    delta_sum = np.zeros((h, w), dtype=np.float32)
    delta_count = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h - win_size + 1, stride):
        for x in range(0, w - win_size + 1, stride):
            source_occ = occlude_patch(source_img, x, y, win_size, win_size, fill_value=fill_value)
            emb_occ = extract_embedding_from_aligned(source_occ)
            sim_occ = cosine_similarity(emb_occ, emb_ref)

            # signed delta
            delta = base_similarity - sim_occ

            delta_sum[y:y+win_size, x:x+win_size] += delta
            delta_count[y:y+win_size, x:x+win_size] += 1.0

    delta_mean = delta_sum / np.maximum(delta_count, 1e-8)
    max_abs = np.max(np.abs(delta_mean)) + 1e-8
    heatmap_signed = delta_mean / max_abs

    return {
        "base_similarity": base_similarity,
        "delta_mean": delta_mean,
        "heatmap_signed": heatmap_signed,
        "win_size": win_size,
        "stride": stride,
        "fill_value": fill_value,
    }

# -------------------------
# 4) Calcular ambos sentidos
# -------------------------
params = {
    "win_size": 16,
    "stride": 4,
    "fill_value": 127,
}

occ_a_vs_b = compute_signed_occlusion_map(
    source_img=aligned_a,
    reference_img=aligned_b,
    **params
)

occ_b_vs_a = compute_signed_occlusion_map(
    source_img=aligned_b,
    reference_img=aligned_a,
    **params
)

# -------------------------
# 5) Reorientar mapa de B al sistema de A
# -------------------------
# Como ambos rostros están alineados al mismo tamaño/canon,
# una aproximación simple y útil es remapear por resize.
heatmap_b_in_a = cv2.resize(
    occ_b_vs_a["heatmap_signed"],
    (aligned_a.shape[1], aligned_a.shape[0]),
    interpolation=cv2.INTER_LINEAR
)

# Promedio de ambos mapas en el sistema de A
heatmap_combined = (occ_a_vs_b["heatmap_signed"] + heatmap_b_in_a) / 2.0

# Renormalización final a [-1, 1]
max_abs_combined = np.max(np.abs(heatmap_combined)) + 1e-8
heatmap_combined = heatmap_combined / max_abs_combined




# %%
# -------------------------
# 6) Overlays
# -------------------------
alpha = 0.15  # ajustar si quieres más/menos transparencia

overlay_a = overlay_signed_heatmap(aligned_a, occ_a_vs_b["heatmap_signed"], alpha=alpha)
overlay_b = overlay_signed_heatmap(aligned_b, occ_b_vs_a["heatmap_signed"], alpha=alpha)
overlay_combined_a = overlay_signed_heatmap(aligned_a, heatmap_combined, alpha=alpha)
overlay_combined_b = overlay_signed_heatmap(aligned_b, heatmap_combined, alpha=alpha)

# -------------------------
# 7) Guardar resultados
# -------------------------
selected_pair["occlusion_A_vs_B_signed"] = occ_a_vs_b
selected_pair["occlusion_B_vs_A_signed"] = occ_b_vs_a
selected_pair["occlusion_combined_signed"] = {
    "heatmap_signed": heatmap_combined,
    "overlay": overlay_combined,
    "alpha": alpha,
    "source": "mean(A_vs_B, remapped_B_vs_A)"
}

# -------------------------
# 8) Mostrar resumen
# -------------------------
print("=== BASE COSINE SIMILARITY ===")
print(f"A vs B: {occ_a_vs_b['base_similarity']:.4f}")
print(f"B vs A: {occ_b_vs_a['base_similarity']:.4f}")
# -------------------------
# 9) Visualización
# -------------------------
fig, axes = plt.subplots(2, 3, figsize=(13, 8))

axes[0, 0].imshow(aligned_a)
axes[0, 0].set_title("Face A")
axes[0, 0].axis("off")

axes[0, 1].imshow(signed_to_color_map(occ_a_vs_b["heatmap_signed"]))
axes[0, 1].set_title("Heatmap A vs B")
axes[0, 1].axis("off")

axes[0, 2].imshow(overlay_a)
axes[0, 2].set_title(f"Overlay A vs B (alpha={alpha:.2f})")
axes[0, 2].axis("off")

axes[1, 0].imshow(aligned_b)
axes[1, 0].set_title("Face B")
axes[1, 0].axis("off")

axes[1, 1].imshow(signed_to_color_map(occ_b_vs_a["heatmap_signed"]))
axes[1, 1].set_title("Heatmap B vs A")
axes[1, 1].axis("off")

axes[1, 2].imshow(overlay_b)
axes[1, 2].set_title(f"Overlay B vs A (alpha={alpha:.2f})")
axes[1, 2].axis("off")


fig2, axes2 = plt.subplots(2, 2, figsize=(13, 8))
axes2[0, 0].imshow(overlay_combined_a)
axes2[0, 0].set_title("overlay_combined_a")
axes2[0, 0].axis("off")

axes2[0, 1].imshow(overlay_combined_b)
axes2[0, 1].set_title("overlay_combined_b")
axes2[0, 1].axis("off")
'''
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 5))
plt.imshow(overlay_combined_a)
plt.title(f"Combined signed map on A (alpha={alpha:.2f})")
plt.axis("off")
plt.show()
'''
