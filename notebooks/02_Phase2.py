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
img1_path = Path("../data/input/img/BrunoFondoBlanco.jpeg")
img2_path = Path("../data/input/img/mateoFotoTarjetaTransporte.jpeg")

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
# VERSION: v1.1
# =========================================
# Descripción:
# - Detecta todos los rostros en varias imágenes
# - Permite expandir la bbox detectada automáticamente
#   con márgenes configurables en X e Y
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
# cargadas como np.ndarray en RGB

import numpy as np
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis

# -------------------------
# 1) Parámetros configurables
# -------------------------
# Márgenes extra relativos al tamaño de la bbox detectada
# Ejemplo:
# - pad_x=0.20 añade 20% del ancho a izquierda y derecha
# - pad_y=0.35 añade 35% del alto arriba y abajo
FACE_PAD_X = 0.20
FACE_PAD_Y = 0.35

# -------------------------
# 2) Inicializar detector
# -------------------------
app = FaceAnalysis(providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

# -------------------------
# 3) Helper: expandir bbox
# -------------------------
def expand_bbox(bbox, image_shape, pad_x=0.20, pad_y=0.35):
    """
    Expande una bounding box facial sin salirse de la imagen.

    Args:
        bbox: iterable (x1, y1, x2, y2)
        image_shape: shape de la imagen (H, W, C)
        pad_x: margen horizontal relativo al ancho de la bbox
        pad_y: margen vertical relativo al alto de la bbox

    Returns:
        tuple: (x1, y1, x2, y2) expandida
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

# -------------------------
# 4) Función de detección
# -------------------------
def detect_faces_in_image(
    img_rgb: np.ndarray,
    photo_label: str,
    pad_x: float = 0.20,
    pad_y: float = 0.35
):
    """
    Detecta todos los rostros de una imagen y devuelve:
    - image_annotated: imagen con cajas e IDs
    - face_records: lista de dicts con metadatos por rostro

    Args:
        img_rgb: imagen RGB
        photo_label: etiqueta de la foto (ej. "F1")
        pad_x: margen horizontal relativo
        pad_y: margen vertical relativo
    """
    faces = app.get(img_rgb)

    image_annotated = img_rgb.copy()
    face_records = []

    # Ordenar de izquierda a derecha para tener IDs reproducibles
    faces_sorted = sorted(faces, key=lambda f: f.bbox[0])

    for i, face in enumerate(faces_sorted, start=1):
        # Bbox original del detector
        bbox_raw = face.bbox.astype(int)

        # Bbox expandida configurable
        x1, y1, x2, y2 = expand_bbox(
            bbox=bbox_raw,
            image_shape=img_rgb.shape,
            pad_x=pad_x,
            pad_y=pad_y
        )

        face_id = f"{photo_label}_R{i}"

        # Recorte expandido del rostro
        crop = img_rgb[y1:y2, x1:x2].copy()

        record = {
            "face_id": face_id,
            "photo_label": photo_label,
            "bbox": (x1, y1, x2, y2),                 # bbox expandida
            "bbox_raw": tuple(map(int, bbox_raw)),    # bbox original del detector
            "det_score": float(face.det_score),
            "crop_rgb": crop,
            "embedding": face.embedding.copy(),
            "kps": face.kps.copy() if hasattr(face, "kps") else None,
        }
        face_records.append(record)

        # Dibujar caja expandida
        image_annotated[y1:y1+2, x1:x2] = [255, 0, 0]
        image_annotated[y2-2:y2, x1:x2] = [255, 0, 0]
        image_annotated[y1:y2, x1:x1+2] = [255, 0, 0]
        image_annotated[y1:y2, x2-2:x2] = [255, 0, 0]

        print(
            f"{face_id} | foto={photo_label} | "
            f"bbox_raw={tuple(map(int, bbox_raw))} | "
            f"bbox_exp={(x1, y1, x2, y2)} | "
            f"score={face.det_score:.3f}"
        )

    return image_annotated, face_records

# -------------------------
# 5) Detectar en ambas fotos
# -------------------------
images = {
    "F1": img1,
    "F2": img2,
}

all_face_records = []
annotated_images = {}

for photo_label, img_rgb in images.items():
    annotated_img, records = detect_faces_in_image(
        img_rgb=img_rgb,
        photo_label=photo_label,
        pad_x=FACE_PAD_X,
        pad_y=FACE_PAD_Y
    )
    annotated_images[photo_label] = annotated_img
    all_face_records.extend(records)

# -------------------------
# 6) Visualizar resultados
# -------------------------
fig, axes = plt.subplots(1, len(images), figsize=(8 * len(images), 8))

if len(images) == 1:
    axes = [axes]

for ax, (photo_label, img_rgb) in zip(axes, annotated_images.items()):
    ax.imshow(img_rgb)
    ax.set_title(f"{photo_label} - rostros detectados")
    ax.axis("off")

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
# 7) Resumen seleccionable
# -------------------------
print("\nResumen de rostros detectados:")
for rec in all_face_records:
    print(f"- {rec['face_id']}  <-  {rec['photo_label']}")

# Ejemplo de ajuste:
# FACE_PAD_X = 0.20
# FACE_PAD_Y = 0.45
#
# Luego reejecutas este bloque y continúas con FASE1_003

# %%
# =========================================
# ID: FASE1_003
# VERSION: v1.2
# =========================================
# Descripción:
# - Selecciona un par de rostros detectados previamente
# - Alinea ambos usando los 5 keypoints de InsightFace
# - Permite ampliar el encuadre del rostro alineado mediante margen extra
# - Visualiza:
#     1) recorte original
#     2) recorte con keypoints
#     3) rostro alineado
#
# Requisitos:
# - Haber ejecutado el bloque FASE1_002
# - Debe existir `all_face_records`

import cv2
import matplotlib.pyplot as plt
import numpy as np
from insightface.utils import face_align

# -------------------------
# 1) Parámetros configurables
# -------------------------
face_id_a = "F1_R1"
face_id_b = "F2_R1"

# Debe ser múltiplo de 112 o 128 para InsightFace
ALIGN_IMAGE_SIZE = 224

# 0.00 = encuadre ajustado
# 0.10-0.25 = más frente, mentón y contorno
ALIGN_MARGIN_RATIO = 0.18

# -------------------------
# 2) Elegir el par
# -------------------------
face_a = next(r for r in all_face_records if r["face_id"] == face_id_a)
face_b = next(r for r in all_face_records if r["face_id"] == face_id_b)

# -------------------------
# 3) Funciones auxiliares
# -------------------------
def align_face_from_record(
    face_record: dict,
    image_size: int = 224,
    margin_ratio: float = 0.18
) -> np.ndarray:
    """
    Alinea un rostro usando los keypoints detectados por InsightFace,
    dejando margen extra alrededor del rostro alineado.

    Args:
        face_record (dict): registro de rostro generado en FASE1_002
        image_size (int): tamaño de salida del rostro alineado
        margin_ratio (float): margen relativo extra dentro del canvas final

    Returns:
        np.ndarray: rostro alineado en RGB
    """
    crop_rgb = face_record["crop_rgb"]
    kps = face_record["kps"]

    if kps is None:
        raise ValueError(f"El rostro {face_record['face_id']} no tiene keypoints disponibles.")

    if not (0.0 <= margin_ratio < 0.5):
        raise ValueError("margin_ratio debe estar entre 0.0 y < 0.5")

    # Pasar keypoints al sistema local del recorte
    x1, y1, _, _ = face_record["bbox"]
    kps_local = kps.copy().astype(np.float32)
    kps_local[:, 0] -= x1
    kps_local[:, 1] -= y1

    # Obtener la transformación de alineamiento de forma robusta
    est = face_align.estimate_norm(kps_local, image_size=image_size)
    M = est[0] if isinstance(est, tuple) else est
    M = np.asarray(M, dtype=np.float32)

    if M.shape != (2, 3):
        raise ValueError(f"estimate_norm devolvió shape inesperado: {M.shape}. Se esperaba (2, 3).")

    # Reducir la escala para dejar margen dentro del canvas final
    scale = 1.0 - (2.0 * margin_ratio)
    M_adj = M.copy()
    M_adj[:, :2] *= scale

    # Recentrar el contenido
    shift = (image_size * (1.0 - scale)) / 2.0
    M_adj[:, 2] = M_adj[:, 2] * scale + shift

    aligned_rgb = cv2.warpAffine(
        crop_rgb,
        M_adj,
        (image_size, image_size),
        borderMode=cv2.BORDER_REPLICATE
    )

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
    kps_local = kps.copy().astype(np.float32)
    kps_local[:, 0] -= x1
    kps_local[:, 1] -= y1

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))

    # Panel 1: recorte original
    axes[0].imshow(crop_rgb)
    axes[0].set_title(f"{face_record['face_id']}\nrecorte")
    axes[0].axis("off")

    # Panel 2: recorte + keypoints
    axes[1].imshow(crop_rgb)
    axes[1].scatter(kps_local[:, 0], kps_local[:, 1], s=40)
    axes[1].set_title(f"{face_record['face_id']}\nkeypoints")
    axes[1].axis("off")

    # Panel 3: alineado
    axes[2].imshow(aligned_rgb)
    axes[2].set_title(f"{face_record['face_id']}\nalineado")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

# -------------------------
# 4) Alinear ambos rostros
# -------------------------
aligned_a = align_face_from_record(
    face_a,
    image_size=ALIGN_IMAGE_SIZE,
    margin_ratio=ALIGN_MARGIN_RATIO
)

aligned_b = align_face_from_record(
    face_b,
    image_size=ALIGN_IMAGE_SIZE,
    margin_ratio=ALIGN_MARGIN_RATIO
)

# -------------------------
# 5) Visualizar alineación
# -------------------------
plot_face_triplet(face_a, aligned_a)
plot_face_triplet(face_b, aligned_b)

# -------------------------
# 6) Guardar resultados para el siguiente bloque
# -------------------------
selected_pair = {
    "face_a": face_a,
    "face_b": face_b,
    "aligned_a": aligned_a,
    "aligned_b": aligned_b,
}

print(f"Par seleccionado: {face_id_a} vs {face_id_b}")
print(f"Shapes alineados: {aligned_a.shape} vs {aligned_b.shape}")
print(f"ALIGN_IMAGE_SIZE={ALIGN_IMAGE_SIZE} | ALIGN_MARGIN_RATIO={ALIGN_MARGIN_RATIO}")

# %%
# =========================================
# ID: FASE2_001
# VERSION: v1.0
# =========================================
# Descripción:
# - Recalcula landmarks densos sobre los rostros ya alineados
# - Usa MediaPipe Face Mesh sobre `aligned_a` y `aligned_b`
# - Visualiza ambos rostros con landmarks
# - Guarda resultados en `selected_pair` para los siguientes bloques
#
# Requisitos:
# - Haber ejecutado el bloque FASE1_003
# - Deben existir:
#     `aligned_a`, `aligned_b`, `selected_pair`
#
# Instalación si hace falta:
# # !pip install mediapipe
#
# Nota:
# - InsightFace te sirvió para detección + alineación global
# - MediaPipe Face Mesh nos da landmarks densos para definir regiones
#   como ojos, nariz, boca, etc.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

# -------------------------
# 1) Inicializar Face Mesh
# -------------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,   # mejora ojos/labios/iris
    min_detection_confidence=0.5
)

# -------------------------
# 2) Función auxiliar:
#    obtener landmarks en píxeles
# -------------------------
def get_face_mesh_landmarks(image_rgb: np.ndarray) -> np.ndarray:
    """
    Ejecuta MediaPipe Face Mesh sobre una imagen RGB y devuelve
    los landmarks en coordenadas de píxel.

    Args:
        image_rgb (np.ndarray): imagen RGB

    Returns:
        np.ndarray: array de shape (N, 2) con coordenadas (x, y)
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

# -------------------------
# 3) Función auxiliar:
#    visualizar landmarks
# -------------------------
def plot_face_with_landmarks(image_rgb: np.ndarray, landmarks: np.ndarray, title: str):
    """
    Muestra una imagen con landmarks superpuestos.
    """
    plt.figure(figsize=(4, 4))
    plt.imshow(image_rgb)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=3)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# -------------------------
# 4) Obtener landmarks densos
# -------------------------
landmarks_a = get_face_mesh_landmarks(aligned_a)
landmarks_b = get_face_mesh_landmarks(aligned_b)

# -------------------------
# 5) Visualizar resultado
# -------------------------
plot_face_with_landmarks(aligned_a, landmarks_a, "aligned_a + landmarks")
plot_face_with_landmarks(aligned_b, landmarks_b, "aligned_b + landmarks")

# -------------------------
# 6) Guardar para bloques siguientes
# -------------------------
selected_pair["landmarks_a"] = landmarks_a
selected_pair["landmarks_b"] = landmarks_b

print("Landmarks densos calculados correctamente.")
print(f"aligned_a: {aligned_a.shape} | landmarks_a: {landmarks_a.shape}")
print(f"aligned_b: {aligned_b.shape} | landmarks_b: {landmarks_b.shape}")

# %%
# =========================================
# ID: FASE2_002
# VERSION: v1.0
# =========================================
# Descripción:
# - Define regiones faciales sobre los rostros alineados
# - Extrae recortes de:
#     1) ojo izquierdo
#     2) ojo derecho
#     3) nariz
#     4) boca
# - Visualiza los recortes de A vs B
# - Guarda todo en `selected_pair["regions"]`
#
# Requisitos:
# - Haber ejecutado FASE2_001
# - Deben existir:
#     `selected_pair["aligned_a"]`
#     `selected_pair["aligned_b"]`
#     `selected_pair["landmarks_a"]`
#     `selected_pair["landmarks_b"]`

import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

# -------------------------
# 1) Helpers de índices
# -------------------------
mp_face_mesh = mp.solutions.face_mesh

def connection_set_to_index_list(connection_set):
    """
    Convierte un set de conexiones de MediaPipe en una lista ordenada de índices únicos.
    """
    idx = set()
    for a, b in connection_set:
        idx.add(a)
        idx.add(b)
    return sorted(idx)

# Índices definidos por MediaPipe
LEFT_EYE_IDX  = connection_set_to_index_list(mp_face_mesh.FACEMESH_LEFT_EYE)
RIGHT_EYE_IDX = connection_set_to_index_list(mp_face_mesh.FACEMESH_RIGHT_EYE)
LIPS_IDX      = connection_set_to_index_list(mp_face_mesh.FACEMESH_LIPS)

# Nariz: subconjunto manual, suficiente para un primer prototipo
NOSE_IDX = sorted(set([
    1, 2, 4, 5, 6, 19, 20, 45, 48, 49, 64, 94, 97, 98,
    115, 122, 129, 168, 195, 197, 218, 275, 278, 279, 294, 327, 331, 344
]))

# -------------------------
# 2) Helpers geométricos
# -------------------------
def get_region_bbox(landmarks: np.ndarray, idx_list: list[int], image_shape: tuple, pad: float = 0.20):
    """
    Calcula una bounding box alrededor de una región definida por landmarks.

    Args:
        landmarks (np.ndarray): shape (N, 2)
        idx_list (list[int]): índices de landmarks de la región
        image_shape (tuple): shape de la imagen (H, W, C)
        pad (float): padding relativo alrededor de la región

    Returns:
        tuple: (x1, y1, x2, y2)
    """
    h, w = image_shape[:2]
    pts = landmarks[idx_list]

    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    bw = x_max - x_min
    bh = y_max - y_min

    x1 = max(0, int(round(x_min - bw * pad)))
    y1 = max(0, int(round(y_min - bh * pad)))
    x2 = min(w, int(round(x_max + bw * pad)))
    y2 = min(h, int(round(y_max + bh * pad)))

    return x1, y1, x2, y2


def crop_from_bbox(image_rgb: np.ndarray, bbox: tuple):
    """
    Recorta una región rectangular de la imagen.
    """
    x1, y1, x2, y2 = bbox
    return image_rgb[y1:y2, x1:x2].copy()


def extract_face_regions(image_rgb: np.ndarray, landmarks: np.ndarray):
    """
    Extrae regiones faciales principales.
    """
    region_definitions = {
        "left_eye": LEFT_EYE_IDX,
        "right_eye": RIGHT_EYE_IDX,
        "nose": NOSE_IDX,
        "mouth": LIPS_IDX,
    }

    out = {}

    for region_name, idx_list in region_definitions.items():
        bbox = get_region_bbox(
            landmarks=landmarks,
            idx_list=idx_list,
            image_shape=image_rgb.shape,
            pad=0.25
        )
        crop = crop_from_bbox(image_rgb, bbox)

        out[region_name] = {
            "bbox": bbox,
            "crop_rgb": crop,
            "landmark_idx": idx_list,
        }

    return out

# -------------------------
# 3) Extraer regiones de A y B
# -------------------------
aligned_a = selected_pair["aligned_a"]
aligned_b = selected_pair["aligned_b"]
landmarks_a = selected_pair["landmarks_a"]
landmarks_b = selected_pair["landmarks_b"]

regions_a = extract_face_regions(aligned_a, landmarks_a)
regions_b = extract_face_regions(aligned_b, landmarks_b)

# -------------------------
# 4) Visualización
# -------------------------
region_names = ["left_eye", "right_eye", "nose", "mouth"]

fig, axes = plt.subplots(len(region_names), 2, figsize=(6, 10))

for i, region_name in enumerate(region_names):
    axes[i, 0].imshow(regions_a[region_name]["crop_rgb"])
    axes[i, 0].set_title(f"A - {region_name}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(regions_b[region_name]["crop_rgb"])
    axes[i, 1].set_title(f"B - {region_name}")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.show()

# -------------------------
# 5) Guardar resultados
# -------------------------
selected_pair["regions"] = {
    "A": regions_a,
    "B": regions_b,
}

print("Regiones faciales extraídas correctamente.")
print("Regiones disponibles:", list(selected_pair["regions"]["A"].keys()))

# %%
# =========================================
# ID: FASE2_004
# VERSION: v1.1
# =========================================
# Descripción:
# - Reconstruye las regiones faciales manteniendo las regiones "fiables"
#   con el mismo criterio que funcionó bien antes:
#       * ojos: FACEMESH_LEFT_EYE / RIGHT_EYE
#       * boca: FACEMESH_LIPS
# - Mantiene nariz con lista manual
# - Añade regiones aproximadas nuevas:
#       * cejas, pómulos, mejillas, mentón, frente
# - Ajusta el mentón para que no suba tanto hacia nariz/cuello
# - Recorta todas las regiones y las guarda en:
#       selected_pair["regions_v2"]
#
# Requisitos:
# - Haber ejecutado FASE2_001
# - Debe existir `selected_pair`

import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

# -------------------------
# 1) Helpers de índices
# -------------------------
mp_face_mesh = mp.solutions.face_mesh

def connection_set_to_index_list(connection_set):
    """
    Convierte un set de conexiones de MediaPipe en una lista ordenada
    de índices únicos.
    """
    idx = set()
    for a, b in connection_set:
        idx.add(a)
        idx.add(b)
    return sorted(idx)

# Regiones oficiales MediaPipe (más fiables)
LEFT_EYE_IDX  = connection_set_to_index_list(mp_face_mesh.FACEMESH_LEFT_EYE)
RIGHT_EYE_IDX = connection_set_to_index_list(mp_face_mesh.FACEMESH_RIGHT_EYE)
LIPS_IDX      = connection_set_to_index_list(mp_face_mesh.FACEMESH_LIPS)

# Regiones manuales / aproximadas
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

# -------------------------
# 2) Helpers geométricos
# -------------------------
def get_region_bbox(landmarks: np.ndarray, idx_list: list[int], image_shape: tuple, pad: float = 0.20):
    """
    Calcula una bounding box alrededor de una región definida por landmarks.
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

def crop_from_bbox(image_rgb: np.ndarray, bbox: tuple):
    """
    Recorta una región rectangular de la imagen.
    """
    x1, y1, x2, y2 = bbox
    return image_rgb[y1:y2, x1:x2].copy()

def get_forehead_bbox(landmarks: np.ndarray, image_shape: tuple):
    """
    Frente aproximada:
    - anchura basada en ambas cejas
    - altura estimada hacia arriba desde las cejas
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

def get_chin_bbox_refined(
    landmarks: np.ndarray,
    image_shape: tuple,
    chin_idx: list[int],
    lips_idx: list[int],
    side_pad: float = 0.18,
    bottom_pad: float = 0.10,
    top_offset_from_mouth: float = 0.55
):
    """
    Calcula una bbox refinada para el mentón.

    Idea:
    - laterales: a partir de landmarks del mentón
    - límite superior: algo por debajo de la boca
    - límite inferior: mentón real + poco padding

    top_offset_from_mouth:
    - cuanto más alto, más pequeño queda el recorte del mentón
    """
    h, w = image_shape[:2]

    chin_pts = landmarks[chin_idx]
    lips_pts = landmarks[lips_idx]

    chin_x_min, _, chin_x_max, chin_y_max = (
        chin_pts[:, 0].min(),
        chin_pts[:, 1].min(),
        chin_pts[:, 0].max(),
        chin_pts[:, 1].max()
    )

    lips_y_min = lips_pts[:, 1].min()
    lips_y_max = lips_pts[:, 1].max()

    chin_w = max(1.0, chin_x_max - chin_x_min)
    lips_h = max(1.0, lips_y_max - lips_y_min)
    chin_h = max(1.0, chin_y_max - chin_pts[:, 1].min())

    # Límite superior: arranca algo por debajo de la boca
    y1 = lips_y_max - (lips_h * (1.0 - top_offset_from_mouth))

    # Límite inferior: base real del mentón + poco margen
    y2 = chin_y_max + chin_h * bottom_pad

    # Laterales a partir del ancho del mentón
    x1 = chin_x_min - chin_w * side_pad
    x2 = chin_x_max + chin_w * side_pad

    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(w, int(round(x2)))
    y2 = min(h, int(round(y2)))

    return (x1, y1, x2, y2)

# -------------------------
# 3) Constructor de regiones
# -------------------------
def extract_regions_v2(image_rgb: np.ndarray, landmarks: np.ndarray):
    """
    Extrae regiones faciales mezclando:
    - regiones oficiales MediaPipe para ojos y boca
    - regiones manuales para el resto
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
        # Tratamiento especial del mentón
        if region_name == "chin":
            bbox = get_chin_bbox_refined(
                landmarks=landmarks,
                image_shape=image_rgb.shape,
                chin_idx=CHIN_IDX,
                lips_idx=LIPS_IDX,
                side_pad=0.18,
                bottom_pad=0.10,
                top_offset_from_mouth=0.55
            )
            crop = crop_from_bbox(image_rgb, bbox)
            out[region_name] = {
                "bbox": bbox,
                "crop_rgb": crop,
                "landmark_idx": idx_list,
                "source": "approx"
            }
            continue

        # Pads específicos para mejorar encuadre
        if region_name in ["left_eye", "right_eye"]:
            pad = 0.25
        elif region_name == "mouth":
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
            "source": "official" if region_name in ["left_eye", "right_eye", "mouth"] else "approx"
        }

    # Frente aproximada
    forehead_bbox = get_forehead_bbox(landmarks, image_rgb.shape)
    out["forehead"] = {
        "bbox": forehead_bbox,
        "crop_rgb": crop_from_bbox(image_rgb, forehead_bbox),
        "landmark_idx": None,
        "source": "approx"
    }

    return out

# -------------------------
# 4) Extraer regiones A/B
# -------------------------
aligned_a = selected_pair["aligned_a"]
aligned_b = selected_pair["aligned_b"]
landmarks_a = selected_pair["landmarks_a"]
landmarks_b = selected_pair["landmarks_b"]

regions_a = extract_regions_v2(aligned_a, landmarks_a)
regions_b = extract_regions_v2(aligned_b, landmarks_b)

# -------------------------
# 5) Visualización rápida
# -------------------------
region_names = [
    "left_eyebrow", "right_eyebrow",
    "left_eye", "right_eye",
    "left_cheekbone", "right_cheekbone",
    "left_cheek", "right_cheek",
    "nose", "mouth", "chin", "forehead"
]

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

# -------------------------
# 6) Guardar resultados
# -------------------------
selected_pair["regions_v2"] = {
    "A": regions_a,
    "B": regions_b,
}

print("Regiones reconstruidas correctamente.")
print("Oficiales:", [k for k, v in regions_a.items() if v["source"] == "official"])
print("Aproximadas:", [k for k, v in regions_a.items() if v["source"] == "approx"])
