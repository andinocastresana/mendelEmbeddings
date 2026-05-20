# =========================================
# ID: PHYLOFACE_EXPERIMENTAL_001
# VERSION: v1.0
# =========================================
# Archivo sugerido:
# phyloface_experimental.py
#
# Objetivo:
# - Unificar funciones rescatadas de Phase1 y Phase2
# - Dejar un único módulo importable para un notebook experimental
# - Cubrir:
#     1) carga
#     2) detección
#     3) selección de par
#     4) alineación
#     5) embeddings globales + métricas
#     6) landmarks densos
#     7) regiones v2
#     8) métricas regionales simples
#
# Nota:
# - Se eligen las versiones mejoradas cuando había duplicados
# - No incluyo aquí occlusion maps; los dejaría aparte por ahora

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from insightface.app import FaceAnalysis
from insightface.utils import face_align


# =========================================================
# 1) CARGA DE IMÁGENES
# =========================================================
# Carga una imagen desde disco usando una ruta str o Path y la devuelve en RGB.
# Hace validación explícita de existencia del archivo y también detecta el caso
# en que OpenCV no logra leerlo aunque el archivo exista.
# No depende de funciones propias del módulo.
def load_image(image_path: str | Path) -> np.ndarray:
    """
    Carga una imagen desde disco y la convierte a RGB.
    """
    image_path = Path(image_path)
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

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# =========================================================
# 2) INSIGHTFACE
# =========================================================
# Inicializa InsightFace FaceAnalysis con el modelo, providers, tamaño de detección
# y contexto indicados. Devuelve el objeto `app` que luego se reutiliza para detectar
# rostros y, más adelante, localizar el submodelo de reconocimiento.
# No depende de funciones propias del módulo.
def init_face_app(
    model_name: str = "buffalo_l",
    det_size: tuple[int, int] = (640, 640),
    ctx_id: int = -1,
    providers: list[str] | None = None,
):
    """
    Inicializa InsightFace FaceAnalysis.
    """
    if providers is None:
        providers = ["CPUExecutionProvider"]

    app = FaceAnalysis(name=model_name, providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    return app

# Expande una bounding box facial original añadiendo margen horizontal y vertical
# relativo al tamaño de la caja, sin salirse de los límites de la imagen.
# Se usa para obtener recortes faciales más amplios y estables que la bbox cruda.
# No depende de funciones propias del módulo.
def expand_bbox(
    bbox,
    image_shape,
    pad_x: float = 0.20,
    pad_y: float = 0.35,
):
    """
    Expande una bbox facial sin salirse de la imagen.
    """
    h, w = image_shape[:2]

    x1, y1, x2, y2 = map(float, bbox)
    bw = x2 - x1
    bh = y2 - y1

    x1_new = max(0, int(round(x1 - bw * pad_x)))
    y1_new = max(0, int(round(y1 - bh * pad_y)))
    x2_new = min(w, int(round(x2 + bw * pad_x)))
    y2_new = min(h, int(round(y2 + bh * pad_y)))

    return (x1_new, y1_new, x2_new, y2_new)

# Detecta todos los rostros de una imagen con InsightFace, los ordena de izquierda
# a derecha para obtener IDs reproducibles, expande cada bbox con `expand_bbox`,
# recorta cada rostro y guarda un registro con metadatos útiles para el resto del flujo:
# face_id, bbox, bbox_raw, det_score, crop_rgb, embedding y kps.
# Devuelve además una copia anotada de la imagen para visualización rápida.
# Depende de: `expand_bbox`.
def detect_faces_in_image(
    app,
    img_rgb: np.ndarray,
    photo_label: str,
    pad_x: float = 0.20,
    pad_y: float = 0.35,
):
    """
    Detecta todos los rostros de una imagen y devuelve:
    - image_annotated
    - face_records
    """
    faces = app.get(img_rgb)

    image_annotated = img_rgb.copy()
    face_records = []

    faces_sorted = sorted(faces, key=lambda f: f.bbox[0])

    for i, face in enumerate(faces_sorted, start=1):
        bbox_raw = face.bbox.astype(int)

        x1, y1, x2, y2 = expand_bbox(
            bbox=bbox_raw,
            image_shape=img_rgb.shape,
            pad_x=pad_x,
            pad_y=pad_y,
        )

        face_id = f"{photo_label}_R{i}"
        crop = img_rgb[y1:y2, x1:x2].copy()

        record = {
            "face_id": face_id,
            "photo_label": photo_label,
            "bbox": (x1, y1, x2, y2),              # bbox expandida
            "bbox_raw": tuple(map(int, bbox_raw)), # bbox original
            "det_score": float(getattr(face, "det_score", np.nan)),
            "crop_rgb": crop,
            "embedding": np.asarray(face.embedding, dtype=np.float32).copy(),
            "kps": face.kps.copy() if hasattr(face, "kps") and face.kps is not None else None,
        }
        face_records.append(record)

        # Dibujar caja expandida
        image_annotated[y1:y1+2, x1:x2] = [255, 0, 0]
        image_annotated[y2-2:y2, x1:x2] = [255, 0, 0]
        image_annotated[y1:y2, x1:x1+2] = [255, 0, 0]
        image_annotated[y1:y2, x2-2:x2] = [255, 0, 0]

    return image_annotated, face_records

# Aplica `detect_faces_in_image` a un conjunto de imágenes etiquetadas, acumulando
# todos los registros faciales en una sola lista y guardando también las imágenes
# anotadas por foto. Es un helper para trabajar cómodamente con varias imágenes
# en el notebook sin repetir código.
# Depende de: `detect_faces_in_image`.
def detect_faces_in_images(
    app,
    images: dict[str, np.ndarray],
    pad_x: float = 0.20,
    pad_y: float = 0.35,
):
    """
    Aplica detección a múltiples imágenes.
    """
    all_face_records = []
    annotated_images = {}

    for photo_label, img_rgb in images.items():
        annotated_img, records = detect_faces_in_image(
            app=app,
            img_rgb=img_rgb,
            photo_label=photo_label,
            pad_x=pad_x,
            pad_y=pad_y,
        )
        annotated_images[photo_label] = annotated_img
        all_face_records.extend(records)

    return annotated_images, all_face_records

# Visualiza las imágenes anotadas tras la detección y superpone sobre cada rostro
# su `face_id`, para facilitar la selección manual del par que se va a comparar.
# No modifica datos; solo sirve como inspección visual del resultado de detección.
# No depende de funciones propias del módulo.
def plot_detected_faces(
    annotated_images: dict[str, np.ndarray],
    all_face_records: list[dict],
):
    """
    Visualiza imágenes anotadas con IDs de rostro.
    """
    fig, axes = plt.subplots(1, len(annotated_images), figsize=(8 * len(annotated_images), 8))

    if len(annotated_images) == 1:
        axes = [axes]

    for ax, (photo_label, img_rgb) in zip(axes, annotated_images.items()):
        ax.imshow(img_rgb)
        ax.set_title(f"{photo_label} - rostros detectados")
        ax.axis("off")

        for rec in all_face_records:
            if rec["photo_label"] == photo_label:
                x1, y1, _, _ = rec["bbox"]
                ax.text(
                    x1,
                    max(10, y1 - 10),
                    rec["face_id"],
                    fontsize=12,
                    bbox=dict(facecolor="yellow", alpha=0.7, edgecolor="black"),
                )

    plt.tight_layout()
    plt.show()

# Recupera de `all_face_records` el diccionario correspondiente a un `face_id`
# concreto. Es una función pequeña pero útil para no repetir búsquedas manuales
# en el notebook y mantener más legible la selección de pares.
# No depende de funciones propias del módulo.
def get_face_record(all_face_records: list[dict], face_id: str) -> dict:
    """
    Recupera un rostro por face_id.
    """
    return next(r for r in all_face_records if r["face_id"] == face_id)


# =========================================================
# 3) ALINEACIÓN
# =========================================================
# Alinea un rostro a partir del recorte y de los 5 keypoints detectados por
# InsightFace. Primero convierte los keypoints al sistema local del recorte,
# luego calcula la transformación de alineación, ajusta su escala para introducir
# un margen configurable dentro del canvas final y genera un rostro alineado
# de tamaño fijo. Esta es la base para estabilizar el par antes de compararlo.
# No depende de funciones propias del módulo.
def align_face_from_record(
    face_record: dict,
    image_size: int = 224,
    margin_ratio: float = 0.18,
) -> np.ndarray:
    """
    Alinea un rostro usando los 5 keypoints de InsightFace,
    dejando margen extra dentro del canvas final.
    """
    crop_rgb = face_record["crop_rgb"]
    kps = face_record["kps"]

    if kps is None:
        raise ValueError(f"El rostro {face_record['face_id']} no tiene keypoints disponibles.")

    if not (0.0 <= margin_ratio < 0.5):
        raise ValueError("margin_ratio debe estar entre 0.0 y < 0.5")

    x1, y1, _, _ = face_record["bbox"]
    kps_local = kps.copy().astype(np.float32)
    kps_local[:, 0] -= x1
    kps_local[:, 1] -= y1

    est = face_align.estimate_norm(kps_local, image_size=image_size)
    M = est[0] if isinstance(est, tuple) else est
    M = np.asarray(M, dtype=np.float32)

    if M.shape != (2, 3):
        raise ValueError(f"estimate_norm devolvió shape inesperado: {M.shape}. Se esperaba (2, 3).")

    scale = 1.0 - (2.0 * margin_ratio)
    M_adj = M.copy()
    M_adj[:, :2] *= scale

    shift = (image_size * (1.0 - scale)) / 2.0
    M_adj[:, 2] = M_adj[:, 2] * scale + shift

    aligned_rgb = cv2.warpAffine(
        crop_rgb,
        M_adj,
        (image_size, image_size),
        borderMode=cv2.BORDER_REPLICATE,
    )

    return aligned_rgb

# Construye la estructura central de trabajo del notebook (`selected_pair`) a partir
# de dos IDs de rostro. Recupera ambos registros con `get_face_record`, los alinea
# con `align_face_from_record` y devuelve un diccionario con face_a, face_b,
# aligned_a, aligned_b y los parámetros de alineación usados.
# Depende de: `get_face_record`, `align_face_from_record`.
def build_selected_pair(
    all_face_records: list[dict],
    face_id_a: str,
    face_id_b: str,
    align_size: int = 224,
    margin_ratio: float = 0.18,
):
    """
    Construye selected_pair a partir de dos face_id.
    """
    face_a = get_face_record(all_face_records, face_id_a)
    face_b = get_face_record(all_face_records, face_id_b)

    aligned_a = align_face_from_record(face_a, image_size=align_size, margin_ratio=margin_ratio)
    aligned_b = align_face_from_record(face_b, image_size=align_size, margin_ratio=margin_ratio)

    return {
        "face_a": face_a,
        "face_b": face_b,
        "aligned_a": aligned_a,
        "aligned_b": aligned_b,
        "align_size": align_size,
        "align_margin_ratio": margin_ratio,
    }

# Muestra, para un rostro concreto, tres vistas complementarias: el recorte original,
# el recorte con keypoints superpuestos y el resultado alineado. Sirve para validar
# visualmente si la detección y la alineación están funcionando correctamente antes
# de pasar a las métricas globales o regionales.
# No depende de funciones propias del módulo.
def plot_face_triplet(face_record: dict, aligned_rgb: np.ndarray):
    """
    Muestra recorte, keypoints y alineado.
    """
    crop_rgb = face_record["crop_rgb"]
    kps = face_record["kps"]

    x1, y1, _, _ = face_record["bbox"]
    kps_local = kps.copy().astype(np.float32)
    kps_local[:, 0] -= x1
    kps_local[:, 1] -= y1

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))

    axes[0].imshow(crop_rgb)
    axes[0].set_title(f"{face_record['face_id']}\nrecorte")
    axes[0].axis("off")

    axes[1].imshow(crop_rgb)
    axes[1].scatter(kps_local[:, 0], kps_local[:, 1], s=40)
    axes[1].set_title(f"{face_record['face_id']}\nkeypoints")
    axes[1].axis("off")

    axes[2].imshow(aligned_rgb)
    axes[2].set_title(f"{face_record['face_id']}\nalineado")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


# =========================================================
# 4) MÉTRICAS GLOBALES
# =========================================================
# Recorre los submodelos internos de FaceAnalysis y devuelve el que expone `get_feat`,
# es decir, el submodelo de reconocimiento facial encargado de generar embeddings.
# Esto permite recalcular embeddings sobre rostros ya alineados en vez de depender
# únicamente del embedding obtenido durante la detección inicial.
# No depende de funciones propias del módulo.
def get_recognition_model(face_app):
    """
    Recupera el submodelo de reconocimiento dentro de FaceAnalysis.
    """
    for _, model in face_app.models.items():
        if hasattr(model, "get_feat"):
            return model
    raise RuntimeError("No se encontró un modelo de reconocimiento con método get_feat().")

# Normaliza un vector a norma L2 = 1. Es la base para que las comparaciones de
# embeddings sean consistentes y comparables entre sí, especialmente para similitud
# coseno y distancia euclídea sobre embeddings faciales.
# No depende de funciones propias del módulo.
def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """
    Normaliza un vector a norma L2 = 1.
    """
    vec = np.asarray(vec, dtype=np.float32).ravel()
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

# Calcula la similitud coseno entre dos vectores tras normalizarlos con `l2_normalize`.
# Es una de las métricas globales principales del flujo y cuanto mayor sea el valor,
# mayor parecido facial sugiere.
# Depende de: `l2_normalize`.
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Similitud coseno.
    """
    v1 = l2_normalize(vec1)
    v2 = l2_normalize(vec2)
    return float(np.dot(v1, v2))

# Calcula la distancia coseno como `1 - cosine_similarity`. Se incluye porque a veces
# resulta más cómodo trabajar con una métrica de distancia que con una de similitud.
# Cuanto menor sea el valor, mayor parecido facial.
# Depende de: `cosine_similarity`.
def cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Distancia coseno.
    """
    return 1.0 - cosine_similarity(vec1, vec2)

# Calcula la distancia euclídea entre dos vectores tras normalizarlos con
# `l2_normalize`. Se usa como métrica complementaria a la similitud coseno para
# comparar embeddings faciales a nivel global.
# Depende de: `l2_normalize`.
def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Distancia euclídea entre embeddings normalizados.
    """
    v1 = l2_normalize(vec1)
    v2 = l2_normalize(vec2)
    return float(np.linalg.norm(v1 - v2))

# Extrae un embedding facial a partir de un rostro ya alineado usando el submodelo
# de reconocimiento recuperado previamente. Permite recalcular embeddings en una
# geometría facial más estable y comparable que la del recorte original.
# No depende de funciones propias del módulo.
def extract_embedding_from_aligned(rec_model, aligned_rgb: np.ndarray) -> np.ndarray:
    """
    Extrae embedding desde un rostro ya alineado.
    """
    emb = rec_model.get_feat(aligned_rgb).flatten()
    return emb.astype(np.float32)

# Ejecuta toda la comparación global del par. Recupera el modelo de reconocimiento,
# toma los embeddings originales guardados en detección, recalcula embeddings sobre
# aligned_a y aligned_b, calcula métricas de control de calidad (estabilidad del
# embedding antes vs después de alinear) y finalmente calcula los scores globales
# principales del par. Todo se guarda dentro de `selected_pair`.
# Depende de: `get_recognition_model`, `extract_embedding_from_aligned`,
#             `cosine_similarity`, `cosine_distance`, `euclidean_distance`.
def compute_global_metrics(app, selected_pair: dict) -> dict:
    """
    Calcula embeddings post-align, QC y métricas globales del par.
    """
    rec_model = get_recognition_model(app)

    face_a = selected_pair["face_a"]
    face_b = selected_pair["face_b"]
    aligned_a = selected_pair["aligned_a"]
    aligned_b = selected_pair["aligned_b"]

    emb_a_original = np.asarray(face_a["embedding"], dtype=np.float32).ravel()
    emb_b_original = np.asarray(face_b["embedding"], dtype=np.float32).ravel()

    emb_a_post = extract_embedding_from_aligned(rec_model, aligned_a)
    emb_b_post = extract_embedding_from_aligned(rec_model, aligned_b)

    qc = {
        "self_similarity_a_original_vs_post_align": cosine_similarity(emb_a_original, emb_a_post),
        "self_similarity_b_original_vs_post_align": cosine_similarity(emb_b_original, emb_b_post),
        "pair_similarity_original": cosine_similarity(emb_a_original, emb_b_original),
        "pair_similarity_post_align": cosine_similarity(emb_a_post, emb_b_post),
    }
    qc["pair_similarity_delta"] = qc["pair_similarity_post_align"] - qc["pair_similarity_original"]

    scores = {
        "cosine_similarity_original": cosine_similarity(emb_a_original, emb_b_original),
        "cosine_similarity_post_align": cosine_similarity(emb_a_post, emb_b_post),
        "cosine_distance_original": cosine_distance(emb_a_original, emb_b_original),
        "cosine_distance_post_align": cosine_distance(emb_a_post, emb_b_post),
        "euclidean_distance_original": euclidean_distance(emb_a_original, emb_b_original),
        "euclidean_distance_post_align": euclidean_distance(emb_a_post, emb_b_post),
    }

    selected_pair["embedding_a_original"] = emb_a_original
    selected_pair["embedding_b_original"] = emb_b_original
    selected_pair["embedding_a_post_align"] = emb_a_post
    selected_pair["embedding_b_post_align"] = emb_b_post
    selected_pair["embedding_qc"] = qc
    selected_pair["global_scores"] = scores

    return selected_pair


# =========================================================
# 5) LANDMARKS DENSOS
# =========================================================
# Inicializa MediaPipe Face Mesh con parámetros adecuados para imágenes estáticas.
# Es el punto de entrada para obtener landmarks densos, que luego se usan para
# definir regiones anatómicas más finas del rostro.
# No depende de funciones propias del módulo.
def init_face_mesh(
    static_image_mode: bool = True,
    max_num_faces: int = 1,
    refine_landmarks: bool = True,
    min_detection_confidence: float = 0.5,
):
    """
    Inicializa MediaPipe Face Mesh.
    """
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        static_image_mode=static_image_mode,
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
    )

# Ejecuta MediaPipe Face Mesh sobre una imagen RGB y devuelve los landmarks faciales
# densos en coordenadas de píxel. Si no detecta una cara, lanza error. Esta función
# es la base para pasar del análisis global a la definición de regiones locales.
# No depende de funciones propias del módulo.
def get_face_mesh_landmarks(face_mesh, image_rgb: np.ndarray) -> np.ndarray:
    """
    Ejecuta Face Mesh y devuelve landmarks en coordenadas de píxel.
    """
    h, w = image_rgb.shape[:2]
    result = face_mesh.process(image_rgb)

    if not result.multi_face_landmarks:
        raise ValueError("No se detectaron landmarks faciales con MediaPipe.")

    face_landmarks = result.multi_face_landmarks[0]
    points = []

    for lm in face_landmarks.landmark:
        x = lm.x * w
        y = lm.y * h
        points.append([x, y])

    return np.array(points, dtype=np.float32)

# Calcula landmarks densos para `aligned_a` y `aligned_b` y los añade a
# `selected_pair` como `landmarks_a` y `landmarks_b`. Deja preparado el par para
# la extracción de regiones anatómicas y el análisis regional posterior.
# Depende de: `get_face_mesh_landmarks`.
def add_dense_landmarks_to_pair(face_mesh, selected_pair: dict) -> dict:
    """
    Añade landmarks densos a selected_pair.
    """
    aligned_a = selected_pair["aligned_a"]
    aligned_b = selected_pair["aligned_b"]

    selected_pair["landmarks_a"] = get_face_mesh_landmarks(face_mesh, aligned_a)
    selected_pair["landmarks_b"] = get_face_mesh_landmarks(face_mesh, aligned_b)

    return selected_pair

# Visualiza una imagen facial junto con sus landmarks densos superpuestos. Sirve
# como control visual para comprobar si MediaPipe está localizando correctamente
# ojos, boca, nariz, cejas y contorno antes de recortar regiones.
# No depende de funciones propias del módulo.
def plot_face_with_landmarks(image_rgb: np.ndarray, landmarks: np.ndarray, title: str):
    """
    Visualiza landmarks sobre una imagen.
    """
    plt.figure(figsize=(4, 4))
    plt.imshow(image_rgb)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=3)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# =========================================================
# 6) REGIONES V2
# =========================================================
mp_face_mesh = mp.solutions.face_mesh

# Convierte un conjunto de conexiones de MediaPipe Face Mesh en una lista ordenada
# de índices únicos. Se usa para derivar, de forma compacta, los índices faciales
# asociados a regiones oficiales como ojos o labios.
# No depende de funciones propias del módulo.
def connection_set_to_index_list(connection_set):
    """
    Convierte conexiones MediaPipe a lista ordenada de índices únicos.
    """
    idx = set()
    for a, b in connection_set:
        idx.add(a)
        idx.add(b)
    return sorted(idx)


LEFT_EYE_IDX = connection_set_to_index_list(mp_face_mesh.FACEMESH_LEFT_EYE)
RIGHT_EYE_IDX = connection_set_to_index_list(mp_face_mesh.FACEMESH_RIGHT_EYE)
LIPS_IDX = connection_set_to_index_list(mp_face_mesh.FACEMESH_LIPS)

NOSE_IDX = sorted(set([
    1, 2, 4, 5, 6, 19, 20, 45, 48, 49, 64, 94, 97, 98,
    115, 122, 129, 168, 195, 197, 218, 275, 278, 279, 294, 327, 331, 344
]))

LEFT_EYEBROW_IDX = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW_IDX = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]

LEFT_CHEEKBONE_IDX = [50, 101, 100, 126, 142, 203, 206]
RIGHT_CHEEKBONE_IDX = [280, 330, 329, 355, 371, 423, 426]

LEFT_CHEEK_IDX = [116, 117, 118, 119, 120, 100, 126, 142, 203, 205, 50]
RIGHT_CHEEK_IDX = [345, 346, 347, 348, 349, 329, 355, 371, 423, 425, 280]

CHIN_IDX = [152, 148, 176, 149, 150, 136, 172, 58, 132, 361, 397, 378, 379, 365, 288, 435]

# Calcula una bounding box rectangular alrededor de una región definida por un
# subconjunto de landmarks, añadiendo un padding configurable y respetando los
# límites de la imagen. Es la base geométrica para recortar regiones locales.
# No depende de funciones propias del módulo.
def get_region_bbox(
    landmarks: np.ndarray,
    idx_list: list[int],
    image_shape: tuple,
    pad: float = 0.20,
):
    """
    Calcula una bbox alrededor de una región definida por landmarks.
    """
    h, w = image_shape[:2]
    pts = landmarks[idx_list]

    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    bw = max(1.0, x_max - x_min)
    bh = max(1.0, y_max - y_min)

    x1 = max(0, int(round(x_min - bw * pad)))
    y1 = max(0, int(round(y_min - bh * pad)))
    x2 = min(w, int(round(x_max + bw * pad)))
    y2 = min(h, int(round(y_max + bh * pad)))

    return (x1, y1, x2, y2)

# Extrae un recorte rectangular de una imagen RGB a partir de una bbox. Es una
# utilidad simple usada repetidamente durante la extracción de regiones faciales.
# No depende de funciones propias del módulo.
def crop_from_bbox(image_rgb: np.ndarray, bbox: tuple):
    """
    Recorta una región rectangular.
    """
    x1, y1, x2, y2 = bbox
    return image_rgb[y1:y2, x1:x2].copy()

# Estima una bbox aproximada para la frente utilizando como referencia la posición
# de ambas cejas y la distancia vertical entre cejas y ojos. Se usa porque la frente
# no viene bien definida como región oficial cerrada en Face Mesh.
# No depende de funciones propias del módulo.
def get_forehead_bbox(landmarks: np.ndarray, image_shape: tuple):
    """
    Frente aproximada.
    """
    h, w = image_shape[:2]

    brow_idx = LEFT_EYEBROW_IDX + RIGHT_EYEBROW_IDX
    eye_idx = LEFT_EYE_IDX + RIGHT_EYE_IDX

    brow_pts = landmarks[brow_idx]
    eye_pts = landmarks[eye_idx]

    x_min, y_min = brow_pts.min(axis=0)
    x_max, _ = brow_pts.max(axis=0)

    brow_center_y = brow_pts[:, 1].mean()
    eye_center_y = eye_pts[:, 1].mean()

    delta = max(8, int(round(abs(eye_center_y - brow_center_y) * 2.0)))

    x1 = max(0, int(round(x_min - 8)))
    x2 = min(w, int(round(x_max + 8)))
    y2 = max(0, int(round(y_min + 2)))
    y1 = max(0, y2 - delta)

    return (x1, y1, x2, y2)

# Calcula una bbox refinada para el mentón usando landmarks del mentón y de los
# labios. Ajusta específicamente el límite superior, inferior y los laterales para
# evitar que el recorte invada demasiado boca, nariz o cuello. Es una versión más
# anatómicamente útil que una bbox simple por min/max.
# No depende de funciones propias del módulo.
def get_chin_bbox_refined(
    landmarks: np.ndarray,
    image_shape: tuple,
    chin_idx: list[int],
    lips_idx: list[int],
    side_pad: float = 0.18,
    bottom_pad: float = 0.10,
    top_offset_from_mouth: float = 0.55,
):
    """
    Bbox refinada para mentón.
    """
    h, w = image_shape[:2]

    chin_pts = landmarks[chin_idx]
    lips_pts = landmarks[lips_idx]

    chin_x_min, _, chin_x_max, chin_y_max = (
        chin_pts[:, 0].min(),
        chin_pts[:, 1].min(),
        chin_pts[:, 0].max(),
        chin_pts[:, 1].max(),
    )

    lips_y_min = lips_pts[:, 1].min()
    lips_y_max = lips_pts[:, 1].max()

    chin_w = max(1.0, chin_x_max - chin_x_min)
    lips_h = max(1.0, lips_y_max - lips_y_min)
    chin_h = max(1.0, chin_y_max - chin_pts[:, 1].min())

    y1 = lips_y_max - (lips_h * (1.0 - top_offset_from_mouth))
    y2 = chin_y_max + chin_h * bottom_pad

    x1 = chin_x_min - chin_w * side_pad
    x2 = chin_x_max + chin_w * side_pad

    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(w, int(round(x2)))
    y2 = min(h, int(round(y2)))

    return (x1, y1, x2, y2)

# Extrae un conjunto ampliado de regiones faciales a partir de landmarks densos.
# Mezcla regiones oficiales más fiables (ojos y boca) con regiones manuales o
# aproximadas (nariz, cejas, pómulos, mejillas, mentón y frente). Para cada región
# guarda bbox, crop, índices de landmarks y fuente ("official" o "approx").
# Es la función central del análisis regional.
# Depende de: `get_chin_bbox_refined`, `get_region_bbox`, `crop_from_bbox`,
#             `get_forehead_bbox`.
def extract_regions_v2(image_rgb: np.ndarray, landmarks: np.ndarray):
    """
    Extrae regiones faciales mezclando regiones oficiales y aproximadas.
    """
    region_defs = {
        "left_eye": LEFT_EYE_IDX,
        "right_eye": RIGHT_EYE_IDX,
        "mouth": LIPS_IDX,
        "nose": NOSE_IDX,
        "left_eyebrow": LEFT_EYEBROW_IDX,
        "right_eyebrow": RIGHT_EYEBROW_IDX,
        "left_cheekbone": LEFT_CHEEKBONE_IDX,
        "right_cheekbone": RIGHT_CHEEKBONE_IDX,
        "left_cheek": LEFT_CHEEK_IDX,
        "right_cheek": RIGHT_CHEEK_IDX,
        "chin": CHIN_IDX,
    }

    out = {}

    for region_name, idx_list in region_defs.items():
        if region_name == "chin":
            bbox = get_chin_bbox_refined(
                landmarks=landmarks,
                image_shape=image_rgb.shape,
                chin_idx=CHIN_IDX,
                lips_idx=LIPS_IDX,
                side_pad=0.18,
                bottom_pad=0.10,
                top_offset_from_mouth=0.55,
            )
            crop = crop_from_bbox(image_rgb, bbox)
            out[region_name] = {
                "bbox": bbox,
                "crop_rgb": crop,
                "landmark_idx": idx_list,
                "source": "approx",
            }
            continue

        if region_name in ["left_eye", "right_eye", "mouth"]:
            pad = 0.25
        elif region_name == "nose":
            pad = 0.22
        else:
            pad = 0.20

        bbox = get_region_bbox(landmarks, idx_list, image_rgb.shape, pad=pad)
        crop = crop_from_bbox(image_rgb, bbox)

        out[region_name] = {
            "bbox": bbox,
            "crop_rgb": crop,
            "landmark_idx": idx_list,
            "source": "official" if region_name in ["left_eye", "right_eye", "mouth"] else "approx",
        }

    forehead_bbox = get_forehead_bbox(landmarks, image_rgb.shape)
    out["forehead"] = {
        "bbox": forehead_bbox,
        "crop_rgb": crop_from_bbox(image_rgb, forehead_bbox),
        "landmark_idx": None,
        "source": "approx",
    }

    return out

# Aplica `extract_regions_v2` a ambos rostros alineados del par y guarda el resultado
# en `selected_pair["regions_v2"]` con la estructura separada para A y B. Deja el
# par listo para comparar regiones o visualizarlas.
# Depende de: `extract_regions_v2`.
def add_regions_v2_to_pair(selected_pair: dict) -> dict:
    """
    Extrae regiones v2 de A y B y las añade a selected_pair.
    """
    aligned_a = selected_pair["aligned_a"]
    aligned_b = selected_pair["aligned_b"]
    landmarks_a = selected_pair["landmarks_a"]
    landmarks_b = selected_pair["landmarks_b"]

    regions_a = extract_regions_v2(aligned_a, landmarks_a)
    regions_b = extract_regions_v2(aligned_b, landmarks_b)

    selected_pair["regions_v2"] = {
        "A": regions_a,
        "B": regions_b,
    }

    return selected_pair

# Visualiza, para una lista de regiones, los recortes de A y B lado a lado. Es una
# herramienta de inspección anatómica para comprobar si la segmentación regional
# resulta razonable antes de calcular métricas locales.
# No depende de funciones propias del módulo.
def plot_regions_v2(selected_pair: dict, region_names: list[str] | None = None):
    """
    Visualiza regiones A/B.
    """
    if region_names is None:
        region_names = [
            "left_eyebrow", "right_eyebrow",
            "left_eye", "right_eye",
            "left_cheekbone", "right_cheekbone",
            "left_cheek", "right_cheek",
            "nose", "mouth", "chin", "forehead",
        ]

    regions_a = selected_pair["regions_v2"]["A"]
    regions_b = selected_pair["regions_v2"]["B"]

    fig, axes = plt.subplots(len(region_names), 2, figsize=(6, 24))

    for i, region_name in enumerate(region_names):
        axes[i, 0].imshow(regions_a[region_name]["crop_rgb"])
        axes[i, 0].set_title(f"A - {region_name}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(regions_b[region_name]["crop_rgb"])
        axes[i, 1].set_title(f"B - {region_name}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()


# =========================================================
# 7) MÉTRICAS REGIONALES SIMPLES
# =========================================================
# Redimensiona dos recortes faciales al mismo tamaño fijo, de forma que puedan ser
# comparados directamente con métricas visuales simples. Es un paso previo necesario
# para la comparación regional basada en intensidad.
# No depende de funciones propias del módulo.
def resize_to_match(img_a: np.ndarray, img_b: np.ndarray, size=(64, 64)):
    """
    Redimensiona dos recortes al mismo tamaño.
    """
    a = cv2.resize(img_a, size, interpolation=cv2.INTER_LINEAR)
    b = cv2.resize(img_b, size, interpolation=cv2.INTER_LINEAR)
    return a, b

# Compara dos parches regionales convirtiéndolos primero a escala de grises,
# normalizando intensidades con z-score y calculando luego similitud coseno.
# Es una métrica regional simple, rápida y útil como primer prototipo antes de
# pasar a embeddings locales o descriptores más sofisticados.
# No depende de funciones propias del módulo.
def grayscale_patch_cosine(patch_a_rgb: np.ndarray, patch_b_rgb: np.ndarray) -> float:
    """
    Similitud coseno simple entre parches en gris con z-score.
    """
    gray_a = cv2.cvtColor(patch_a_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32).ravel()
    gray_b = cv2.cvtColor(patch_b_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32).ravel()

    gray_a = (gray_a - gray_a.mean()) / (gray_a.std() + 1e-8)
    gray_b = (gray_b - gray_b.mean()) / (gray_b.std() + 1e-8)

    denom = (np.linalg.norm(gray_a) * np.linalg.norm(gray_b)) + 1e-8
    sim = float(np.dot(gray_a, gray_b) / denom)
    return max(-1.0, min(1.0, sim))

# Calcula una métrica regional simple para todas las regiones comunes entre A y B.
# Toma los recortes de `regions_v2`, los iguala de tamaño con `resize_to_match`,
# calcula una similitud visual con `grayscale_patch_cosine` y guarda los resultados
# en `selected_pair["regional_scores"]`.
# Depende de: `resize_to_match`, `grayscale_patch_cosine`.
def compare_regions_v2(selected_pair: dict, resize_shape=(64, 64)) -> dict:
    """
    Calcula una métrica regional simple por recorte.
    """
    regions_a = selected_pair["regions_v2"]["A"]
    regions_b = selected_pair["regions_v2"]["B"]

    common_regions = sorted(set(regions_a.keys()).intersection(set(regions_b.keys())))
    regional_scores = {}

    for region_name in common_regions:
        crop_a = regions_a[region_name]["crop_rgb"]
        crop_b = regions_b[region_name]["crop_rgb"]

        if crop_a.size == 0 or crop_b.size == 0:
            continue

        patch_a, patch_b = resize_to_match(crop_a, crop_b, size=resize_shape)
        regional_scores[region_name] = {
            "gray_cosine": grayscale_patch_cosine(patch_a, patch_b),
            "shape_a": crop_a.shape,
            "shape_b": crop_b.shape,
        }

    selected_pair["regional_scores"] = regional_scores
    return selected_pair

# Imprime un resumen corto y legible de la comparación global del par: principales
# scores post-align y métricas de control de calidad del embedding. Está pensada
# para dejar el notebook limpio y evitar repetir prints manuales.
# No depende de funciones propias del módulo.
def print_global_summary(selected_pair: dict):
    """
    Resumen corto de métricas globales.
    """
    face_a = selected_pair["face_a"]["face_id"]
    face_b = selected_pair["face_b"]["face_id"]
    scores = selected_pair["global_scores"]
    qc = selected_pair["embedding_qc"]

    print("=== COMPARACIÓN GLOBAL ===")
    print(f"{face_a} vs {face_b}")
    print(f"cosine_similarity_post_align : {scores['cosine_similarity_post_align']:.4f}")
    print(f"cosine_distance_post_align   : {scores['cosine_distance_post_align']:.4f}")
    print(f"euclidean_distance_post_align: {scores['euclidean_distance_post_align']:.4f}")
    print()
    print("=== QC EMBEDDING ===")
    print(f"self_similarity_a_original_vs_post_align: {qc['self_similarity_a_original_vs_post_align']:.4f}")
    print(f"self_similarity_b_original_vs_post_align: {qc['self_similarity_b_original_vs_post_align']:.4f}")
    print(f"pair_similarity_delta                  : {qc['pair_similarity_delta']:+.4f}")

# Imprime un resumen corto de la comparación regional, mostrando para cada región
# el score actualmente calculado. Es una función de salida simple para revisar de
# un vistazo qué zonas parecen más o menos similares entre ambos rostros.
# No depende de funciones propias del módulo.
def print_regional_summary(selected_pair: dict):
    """
    Resumen corto de métricas regionales.
    """
    print("=== COMPARACIÓN REGIONAL ===")
    for region_name, metrics in sorted(selected_pair["regional_scores"].items()):
        print(f"{region_name:16s} | gray_cosine={metrics['gray_cosine']:.4f}")



#Posibilidad de calculo de rejiones ajustadas a los puntos
# =========================================
# ID: PHYLOFACE_REGIONS_MASK_001
# VERSION: v1.0
# =========================================
# Añade / reemplaza estas funciones en tu módulo experimental.
#
# Qué añade:
# - máscaras poligonales por región
# - recorte rectangular + recorte enmascarado
# - plotting de regiones rectangulares y enmascaradas
#
# Nota:
# - para regiones "official" usa polygon_idx si existe
# - para regiones aproximadas usa convex hull de landmark_idx
# - se conservan bbox y crop_rgb para compatibilidad


import cv2
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1) CONTORNOS POLIGONALES MANUALES (cuando interesa)
# --------------------------------------------------
LEFT_EYE_POLYGON_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_POLYGON_IDX = [362, 385, 387, 263, 373, 380]

# contorno externo de labios
MOUTH_POLYGON_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

LEFT_EYEBROW_POLYGON_IDX = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW_POLYGON_IDX = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]

# nariz: aproximación razonable para borde externo
NOSE_POLYGON_IDX = [168, 6, 197, 195, 5, 4, 1, 94, 97, 2, 326, 327]


# --------------------------------------------------
# 2) HELPERS DE MÁSCARA / RECORTE
# --------------------------------------------------
def create_region_mask_from_points(
    image_shape: tuple,
    landmarks: np.ndarray,
    polygon_idx: list[int] | None = None,
    landmark_idx: list[int] | None = None,
):
    """
    Genera una máscara binaria 2D para una región facial.

    Prioridad:
    - si se proporciona polygon_idx, usa esos puntos como contorno
    - si no, construye un convex hull a partir de landmark_idx

    Returns:
        mask (H, W) uint8 con valores 0/255
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if polygon_idx is not None and len(polygon_idx) >= 3:
        pts = landmarks[polygon_idx].astype(np.int32)
    elif landmark_idx is not None and len(landmark_idx) >= 3:
        pts = landmarks[landmark_idx].astype(np.int32)
        pts = cv2.convexHull(pts)
    else:
        return mask

    cv2.fillPoly(mask, [pts], 255)
    return mask


def crop_mask_and_image(image_rgb: np.ndarray, mask: np.ndarray, bbox: tuple):
    """
    Recorta simultáneamente:
    - la imagen rectangular
    - la máscara
    - la imagen enmascarada

    Returns:
        crop_rgb
        crop_mask
        crop_masked_rgb
    """
    x1, y1, x2, y2 = bbox

    crop_rgb = image_rgb[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2].copy()

    crop_masked_rgb = crop_rgb.copy()
    crop_masked_rgb[crop_mask == 0] = 0

    return crop_rgb, crop_mask, crop_masked_rgb


def plot_face_regions_overlay(
    image_rgb: np.ndarray,
    regions: dict,
    alpha: float = 0.30,
    title: str = "regions overlay",
):
    """
    Superpone sobre la cara las máscaras de las regiones disponibles.
    """
    overlay = image_rgb.copy()

    color_map = {
        "left_eye": (255, 0, 0),
        "right_eye": (0, 255, 0),
        "mouth": (255, 255, 0),
        "nose": (0, 0, 255),
        "left_eyebrow": (255, 0, 255),
        "right_eyebrow": (0, 255, 255),
        "left_cheekbone": (255, 128, 0),
        "right_cheekbone": (128, 255, 0),
        "left_cheek": (255, 128, 128),
        "right_cheek": (128, 255, 255),
        "chin": (180, 180, 255),
        "forehead": (200, 100, 255),
    }

    for region_name, region_data in regions.items():
        mask = region_data.get("mask")
        if mask is None:
            continue

        color = np.array(color_map.get(region_name, (200, 200, 200)), dtype=np.float32)
        idx = mask > 0
        overlay[idx] = ((1.0 - alpha) * overlay[idx] + alpha * color).astype(np.uint8)

    plt.figure(figsize=(5, 5))
    plt.imshow(overlay)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# 3) EXTRACTOR DE REGIONES V2 CON MÁSCARAS
# --------------------------------------------------
def extract_regions_v2_masked(image_rgb: np.ndarray, landmarks: np.ndarray):
    """
    Extrae regiones faciales v2 guardando:
    - bbox
    - mask global
    - crop_rgb rectangular
    - crop_mask
    - crop_masked_rgb
    - landmark_idx
    - polygon_idx
    - source

    Estrategia:
    - official / anatómicas pequeñas: polygon_idx manual si existe
    - resto: convex hull sobre landmark_idx
    """
    region_defs = {
        "left_eye": {
            "landmark_idx": LEFT_EYE_IDX,
            "polygon_idx": LEFT_EYE_POLYGON_IDX,
            "source": "official",
            "pad": 0.25,
        },
        "right_eye": {
            "landmark_idx": RIGHT_EYE_IDX,
            "polygon_idx": RIGHT_EYE_POLYGON_IDX,
            "source": "official",
            "pad": 0.25,
        },
        "mouth": {
            "landmark_idx": LIPS_IDX,
            "polygon_idx": MOUTH_POLYGON_IDX,
            "source": "official",
            "pad": 0.25,
        },
        "nose": {
            "landmark_idx": NOSE_IDX,
            "polygon_idx": NOSE_POLYGON_IDX,
            "source": "approx",
            "pad": 0.22,
        },
        "left_eyebrow": {
            "landmark_idx": LEFT_EYEBROW_IDX,
            "polygon_idx": LEFT_EYEBROW_POLYGON_IDX,
            "source": "approx",
            "pad": 0.20,
        },
        "right_eyebrow": {
            "landmark_idx": RIGHT_EYEBROW_IDX,
            "polygon_idx": RIGHT_EYEBROW_POLYGON_IDX,
            "source": "approx",
            "pad": 0.20,
        },
        "left_cheekbone": {
            "landmark_idx": LEFT_CHEEKBONE_IDX,
            "polygon_idx": None,
            "source": "approx",
            "pad": 0.20,
        },
        "right_cheekbone": {
            "landmark_idx": RIGHT_CHEEKBONE_IDX,
            "polygon_idx": None,
            "source": "approx",
            "pad": 0.20,
        },
        "left_cheek": {
            "landmark_idx": LEFT_CHEEK_IDX,
            "polygon_idx": None,
            "source": "approx",
            "pad": 0.20,
        },
        "right_cheek": {
            "landmark_idx": RIGHT_CHEEK_IDX,
            "polygon_idx": None,
            "source": "approx",
            "pad": 0.20,
        },
    }

    out = {}

    # regiones estándar
    for region_name, cfg in region_defs.items():
        bbox = get_region_bbox(
            landmarks=landmarks,
            idx_list=cfg["landmark_idx"],
            image_shape=image_rgb.shape,
            pad=cfg["pad"],
        )

        mask = create_region_mask_from_points(
            image_shape=image_rgb.shape,
            landmarks=landmarks,
            polygon_idx=cfg["polygon_idx"],
            landmark_idx=cfg["landmark_idx"],
        )

        crop_rgb, crop_mask, crop_masked_rgb = crop_mask_and_image(
            image_rgb=image_rgb,
            mask=mask,
            bbox=bbox,
        )

        out[region_name] = {
            "bbox": bbox,
            "mask": mask,
            "crop_rgb": crop_rgb,
            "crop_mask": crop_mask,
            "crop_masked_rgb": crop_masked_rgb,
            "landmark_idx": cfg["landmark_idx"],
            "polygon_idx": cfg["polygon_idx"],
            "source": cfg["source"],
        }

    # mentón refinado
    chin_bbox = get_chin_bbox_refined(
        landmarks=landmarks,
        image_shape=image_rgb.shape,
        chin_idx=CHIN_IDX,
        lips_idx=LIPS_IDX,
        side_pad=0.18,
        bottom_pad=0.10,
        top_offset_from_mouth=0.55,
    )
    chin_mask = create_region_mask_from_points(
        image_shape=image_rgb.shape,
        landmarks=landmarks,
        polygon_idx=None,
        landmark_idx=CHIN_IDX,
    )
    chin_crop_rgb, chin_crop_mask, chin_crop_masked_rgb = crop_mask_and_image(
        image_rgb=image_rgb,
        mask=chin_mask,
        bbox=chin_bbox,
    )
    out["chin"] = {
        "bbox": chin_bbox,
        "mask": chin_mask,
        "crop_rgb": chin_crop_rgb,
        "crop_mask": chin_crop_mask,
        "crop_masked_rgb": chin_crop_masked_rgb,
        "landmark_idx": CHIN_IDX,
        "polygon_idx": None,
        "source": "approx",
    }

    # frente aproximada
    forehead_bbox = get_forehead_bbox(landmarks, image_rgb.shape)
    forehead_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = forehead_bbox
    forehead_mask[y1:y2, x1:x2] = 255

    forehead_crop_rgb, forehead_crop_mask, forehead_crop_masked_rgb = crop_mask_and_image(
        image_rgb=image_rgb,
        mask=forehead_mask,
        bbox=forehead_bbox,
    )
    out["forehead"] = {
        "bbox": forehead_bbox,
        "mask": forehead_mask,
        "crop_rgb": forehead_crop_rgb,
        "crop_mask": forehead_crop_mask,
        "crop_masked_rgb": forehead_crop_masked_rgb,
        "landmark_idx": None,
        "polygon_idx": None,
        "source": "approx",
    }

    return out


def add_regions_v2_masked_to_pair(selected_pair: dict) -> dict:
    """
    Extrae regiones v2 con máscara para A y B y las guarda en selected_pair.
    """
    aligned_a = selected_pair["aligned_a"]
    aligned_b = selected_pair["aligned_b"]
    landmarks_a = selected_pair["landmarks_a"]
    landmarks_b = selected_pair["landmarks_b"]

    regions_a = extract_regions_v2_masked(aligned_a, landmarks_a)
    regions_b = extract_regions_v2_masked(aligned_b, landmarks_b)

    selected_pair["regions_v2"] = {
        "A": regions_a,
        "B": regions_b,
    }

    return selected_pair


# --------------------------------------------------
# 4) PLOTTING DE NUEVAS REGIONES
# --------------------------------------------------
def plot_regions_v2_masked(
    selected_pair: dict,
    region_names: list[str] | None = None,
    mode: str = "masked",
):
    """
    Visualiza regiones A/B.

    mode:
    - 'rect'   -> muestra crop_rgb rectangular
    - 'masked' -> muestra crop_masked_rgb
    - 'mask'   -> muestra crop_mask binaria
    """
    if region_names is None:
        region_names = [
            "left_eyebrow", "right_eyebrow",
            "left_eye", "right_eye",
            "left_cheekbone", "right_cheekbone",
            "left_cheek", "right_cheek",
            "nose", "mouth", "chin", "forehead",
        ]

    regions_a = selected_pair["regions_v2"]["A"]
    regions_b = selected_pair["regions_v2"]["B"]

    fig, axes = plt.subplots(len(region_names), 2, figsize=(6, 24))
    if len(region_names) == 1:
        axes = np.array([axes])

    for i, region_name in enumerate(region_names):
        if mode == "rect":
            img_a = regions_a[region_name]["crop_rgb"]
            img_b = regions_b[region_name]["crop_rgb"]
            cmap = None
        elif mode == "mask":
            img_a = regions_a[region_name]["crop_mask"]
            img_b = regions_b[region_name]["crop_mask"]
            cmap = "gray"
        else:
            img_a = regions_a[region_name]["crop_masked_rgb"]
            img_b = regions_b[region_name]["crop_masked_rgb"]
            cmap = None

        axes[i, 0].imshow(img_a, cmap=cmap)
        axes[i, 0].set_title(f"A - {region_name} [{mode}]")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(img_b, cmap=cmap)
        axes[i, 1].set_title(f"B - {region_name} [{mode}]")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()


def plot_region_detail(
    selected_pair: dict,
    side: str,
    region_name: str,
):
    """
    Muestra para una sola región:
    - overlay sobre la cara completa
    - crop rectangular
    - máscara recortada
    - crop enmascarado
    """
    if side not in ["A", "B"]:
        raise ValueError("side debe ser 'A' o 'B'")

    image_rgb = selected_pair["aligned_a"] if side == "A" else selected_pair["aligned_b"]
    region = selected_pair["regions_v2"][side][region_name]

    # overlay simple
    overlay = image_rgb.copy()
    mask = region["mask"]
    overlay[mask > 0] = (0.7 * overlay[mask > 0] + 0.3 * np.array([255, 0, 0])).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    axes[0].imshow(overlay)
    axes[0].set_title(f"{side} - {region_name}\noverlay")
    axes[0].axis("off")

    axes[1].imshow(region["crop_rgb"])
    axes[1].set_title("crop_rgb")
    axes[1].axis("off")

    axes[2].imshow(region["crop_mask"], cmap="gray")
    axes[2].set_title("crop_mask")
    axes[2].axis("off")

    axes[3].imshow(region["crop_masked_rgb"])
    axes[3].set_title("crop_masked_rgb")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# 5) MÉTRICA REGIONAL SIMPLE USANDO MÁSCARA
# --------------------------------------------------
def masked_grayscale_patch_cosine(patch_a_rgb: np.ndarray, patch_b_rgb: np.ndarray) -> float:
    """
    Métrica simple entre recortes enmascarados.
    """
    gray_a = cv2.cvtColor(patch_a_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32).ravel()
    gray_b = cv2.cvtColor(patch_b_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32).ravel()

    valid = ~((gray_a == 0) & (gray_b == 0))
    gray_a = gray_a[valid]
    gray_b = gray_b[valid]

    if len(gray_a) == 0 or len(gray_b) == 0:
        return np.nan

    gray_a = (gray_a - gray_a.mean()) / (gray_a.std() + 1e-8)
    gray_b = (gray_b - gray_b.mean()) / (gray_b.std() + 1e-8)

    denom = (np.linalg.norm(gray_a) * np.linalg.norm(gray_b)) + 1e-8
    sim = float(np.dot(gray_a, gray_b) / denom)
    return max(-1.0, min(1.0, sim))


def compare_regions_v2_masked(selected_pair: dict, resize_shape=(64, 64)) -> dict:
    """
    Compara regiones usando los recortes enmascarados.
    """
    regions_a = selected_pair["regions_v2"]["A"]
    regions_b = selected_pair["regions_v2"]["B"]

    common_regions = sorted(set(regions_a.keys()).intersection(set(regions_b.keys())))
    regional_scores = {}

    for region_name in common_regions:
        crop_a = regions_a[region_name]["crop_masked_rgb"]
        crop_b = regions_b[region_name]["crop_masked_rgb"]

        if crop_a.size == 0 or crop_b.size == 0:
            continue

        patch_a = cv2.resize(crop_a, resize_shape, interpolation=cv2.INTER_LINEAR)
        patch_b = cv2.resize(crop_b, resize_shape, interpolation=cv2.INTER_LINEAR)

        regional_scores[region_name] = {
            "gray_cosine_masked": masked_grayscale_patch_cosine(patch_a, patch_b),
            "shape_a": crop_a.shape,
            "shape_b": crop_b.shape,
        }

    selected_pair["regional_scores"] = regional_scores
    return selected_pair


# --------------------------------------------------
# 6) EJEMPLO DE USO EN NOTEBOOK
# --------------------------------------------------
# selected_pair = add_regions_v2_masked_to_pair(selected_pair)
# plot_face_regions_overlay(selected_pair["aligned_a"], selected_pair["regions_v2"]["A"], title="A overlay")
# plot_face_regions_overlay(selected_pair["aligned_b"], selected_pair["regions_v2"]["B"], title="B overlay")
# plot_regions_v2_masked(selected_pair, mode="rect")
# plot_regions_v2_masked(selected_pair, mode="masked")
# plot_regions_v2_masked(selected_pair, mode="mask")
# plot_region_detail(selected_pair, side="A", region_name="left_eye")
# selected_pair = compare_regions_v2_masked(selected_pair)
# print(selected_pair["regional_scores"])