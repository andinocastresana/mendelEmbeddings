# =========================================
# ID: PHYLOFACE_PAIRS_001
# VERSION: v1.0
# =========================================
# Origen de las funciones (migración Tarea #1, Paso 6c):
#   src/phyloface_experimental_functions.py
#   - init_face_app                (línea 74)
#   - expand_bbox                  (línea 94)
#   - detect_faces_in_image        (línea 122)
#   - detect_faces_in_images       (línea 179)
#   - get_face_record              (línea 243)
#   - align_face_from_record       (línea 259)
#   - build_selected_pair          (línea 310)
#
# Qué hace este módulo:
# - Provee el camino completo "imagen(es) en disco → par seleccionado para
#   comparar". Esto incluye: inicialización de InsightFace, detección de
#   rostros con bbox expandida + crop + embedding inicial + 5 keypoints,
#   selección manual del par a comparar por `face_id`, y alineación
#   canónica de cada rostro a un canvas de tamaño fijo con margen extra.
#
# - La estructura de datos central es el `face_record` (un dict por rostro)
#   y el `selected_pair` (dict con face_a + face_b + aligned_a + aligned_b
#   + parámetros de alineación). Estos contratos los consumen el resto de
#   módulos del paquete (`landmarks`, `regions`, `comparator_global`,
#   `comparator_regional`, `viz`).
#
# Por qué existe en paralelo a `core/detector.py`:
# - `core/detector.py` está orientado al **caché por imagen** y multi-cara:
#   produce un payload pensado para `save_image_cache`. Su estructura
#   (arrays Nx-D, no lista de dicts) no es la que esperan los módulos
#   experimentales migrados.
# - `core/pairs.py` (este archivo) replica el flujo del notebook
#   experimental: lista de dicts `face_records`, selección manual por
#   `face_id` (ej. "F1_R2"), alineación a 224x224 con margen.
# - A futuro convendrá unificar ambos caminos; mientras tanto coexisten
#   sin pisarse.
#
# Decisiones de diseño:
# - Los rostros se **ordenan por x1 ascendente** (izquierda → derecha en
#   la imagen) antes de asignarles ID (`<photo_label>_R<n>`, 1-based).
#   Esto hace que los IDs sean reproducibles y predecibles entre corridas.
# - La bbox guardada en el record es la **expandida**; la cruda se preserva
#   en `bbox_raw` por si alguien la necesita.
# - `align_face_from_record` valida explícitamente la forma de la matriz
#   devuelta por `face_align.estimate_norm` (algunas versiones de InsightFace
#   devuelven tupla, otras matriz; algunas con shape (3,3), otras (2,3)).
#
# Cosas que NO hace este módulo:
# - No visualiza (eso vive en `viz/detection.py`).
# - No calcula métricas (eso vive en `core/comparator_global.py`,
#   `comparator_regional.py`).
# - No re-extrae embeddings sobre la cara alineada (eso es
#   `core/embedder.py`).

# -----------------------------------------
# FILE: phyloface/core/pairs.py
# -----------------------------------------

import cv2
import numpy as np

from insightface.app import FaceAnalysis
from insightface.utils import face_align


# =========================================================
# 1) Inicialización de InsightFace FaceAnalysis
# =========================================================
# Construye una instancia de `FaceAnalysis` configurada con modelo, providers,
# tamaño de detección y contexto. La función espera que el modelo ya esté
# cacheado localmente (InsightFace lo descarga la primera vez).
# Por defecto usa CPU; para GPU pasar `providers=["CUDAExecutionProvider"]`
# y `ctx_id=0` (o el id de GPU correspondiente).
# No depende de funciones propias del módulo.
def init_face_app(
    model_name: str = "buffalo_l",
    det_size: tuple[int, int] = (640, 640),
    ctx_id: int = -1,
    providers: list[str] | None = None,
):
    """
    Inicializa InsightFace FaceAnalysis.

    Parámetros:
        model_name: nombre del bundle de modelos (ej. 'buffalo_l',
            'buffalo_s', 'antelopev2'). Define qué submodelos se cargan.
        det_size: resolución de detección (W, H). Más grande = más caras
            chicas detectadas, más lento.
        ctx_id: -1 para CPU, >=0 para GPU.
        providers: lista de execution providers de ONNX Runtime. Si es
            None, se asume CPU.

    Devuelve:
        Instancia de `FaceAnalysis` ya preparada para usar (`app.get(img)`).
    """
    # Default a CPU si no se pasó providers — evita errores en entornos
    # sin CUDA configurada.
    if providers is None:
        providers = ["CPUExecutionProvider"]

    # Construcción del FaceAnalysis (carga todos los submodelos del bundle).
    app = FaceAnalysis(name=model_name, providers=providers)

    # `prepare` configura el contexto de ejecución y el tamaño de detección.
    # Es donde realmente "calienta" los modelos en memoria.
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    return app


# =========================================================
# 2) Expansión de bbox facial con padding
# =========================================================
# Recibe una bbox cruda y la expande proporcionalmente al tamaño de la
# caja, sin salirse de los límites de la imagen. Útil porque el detector
# devuelve bboxes "justas" — incluir un poco de contexto alrededor
# (frente, mentón, orejas) suele mejorar tanto la alineación como la
# percepción humana del crop.
# No depende de funciones propias del módulo.
def expand_bbox(
    bbox,
    image_shape,
    pad_x: float = 0.20,
    pad_y: float = 0.35,
):
    """
    Expande una bbox facial con padding proporcional, sin salirse de la imagen.

    Parámetros:
        bbox: tupla/array (x1, y1, x2, y2) en píxeles.
        image_shape: shape de la imagen (H, W[, C]) para clipping.
        pad_x: padding horizontal relativo al ancho de la bbox (0.20 = 20%).
        pad_y: padding vertical relativo al alto. Más grande que pad_x por
            defecto porque la cara humana es más alta que ancha (frente +
            mentón).

    Devuelve:
        Tupla (x1, y1, x2, y2) en píxeles, dentro de los límites de la imagen.
    """
    h, w = image_shape[:2]

    # Convertimos a float para no perder precisión en los cálculos
    # intermedios (vamos a redondear al final).
    x1, y1, x2, y2 = map(float, bbox)
    bw = x2 - x1
    bh = y2 - y1

    # Expansión proporcional al tamaño de la caja en cada eje, clipeada
    # a [0, w/h] al convertir a int.
    x1_new = max(0, int(round(x1 - bw * pad_x)))
    y1_new = max(0, int(round(y1 - bh * pad_y)))
    x2_new = min(w, int(round(x2 + bw * pad_x)))
    y2_new = min(h, int(round(y2 + bh * pad_y)))

    return (x1_new, y1_new, x2_new, y2_new)


# =========================================================
# 3) Detección + face_records para una imagen
# =========================================================
# Corre detección sobre la imagen, ordena los rostros de izquierda a
# derecha (para IDs reproducibles), expande la bbox de cada uno, recorta,
# y arma un dict `face_record` por rostro con todos los metadatos que el
# pipeline río abajo va a necesitar.
#
# Devuelve también una **copia anotada de la imagen** con las bboxes
# expandidas dibujadas como recuadros rojos finos, útil para visualizar
# qué encontró el detector antes de elegir el par a comparar.
#
# Depende de: expand_bbox.
def detect_faces_in_image(
    app,
    img_rgb: np.ndarray,
    photo_label: str,
    pad_x: float = 0.20,
    pad_y: float = 0.35,
):
    """
    Detecta rostros en una imagen y arma los face_records.

    Parámetros:
        app: instancia de FaceAnalysis (ver `init_face_app`).
        img_rgb: imagen RGB (H, W, 3).
        photo_label: etiqueta corta de la foto (ej. "F1"). Se usa como
            prefijo del face_id: "F1_R1", "F1_R2", ...
        pad_x, pad_y: padding proporcional para `expand_bbox`.

    Devuelve:
        (image_annotated, face_records)
            image_annotated: copia de img_rgb con bboxes dibujadas.
            face_records: lista de dicts (uno por rostro detectado).
    """
    # La detección puede devolver 0..N rostros.
    faces = app.get(img_rgb)

    # Trabajamos sobre una copia para no mutar la imagen original al dibujar.
    image_annotated = img_rgb.copy()
    face_records = []

    # Orden izquierda → derecha por x1 de la bbox. Esto hace que los IDs
    # ("F1_R1", "F1_R2", ...) sean reproducibles entre corridas — siempre
    # R1 será el más a la izquierda.
    faces_sorted = sorted(faces, key=lambda f: f.bbox[0])

    # Enumeramos desde 1 (los IDs son 1-based, más amigables que 0-based
    # para uso manual desde el notebook).
    for i, face in enumerate(faces_sorted, start=1):
        # bbox cruda del detector (la guardamos como backup en bbox_raw).
        bbox_raw = face.bbox.astype(int)

        # bbox expandida (la que se usa en todo el flujo principal).
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
            "bbox": (x1, y1, x2, y2),               # bbox expandida (la principal)
            "bbox_raw": tuple(map(int, bbox_raw)),  # bbox cruda (referencia)
            "det_score": float(getattr(face, "det_score", np.nan)),
            "crop_rgb": crop,
            # Embeddings y keypoints copiados para que el record sea
            # autocontenido (sobrevive a cambios en el objeto `face`).
            "embedding": np.asarray(face.embedding, dtype=np.float32).copy(),
            "kps": face.kps.copy() if hasattr(face, "kps") and face.kps is not None else None,
        }
        face_records.append(record)

        # Dibuja la bbox expandida sobre la copia anotada (rojo, grosor 2 px).
        # Se hace por "rebanadas" en vez de cv2.rectangle para mantener la
        # implementación idéntica al archivo original.
        image_annotated[y1:y1+2, x1:x2] = [255, 0, 0]    # borde superior
        image_annotated[y2-2:y2, x1:x2] = [255, 0, 0]    # borde inferior
        image_annotated[y1:y2, x1:x1+2] = [255, 0, 0]    # borde izquierdo
        image_annotated[y1:y2, x2-2:x2] = [255, 0, 0]    # borde derecho

    return image_annotated, face_records


# =========================================================
# 4) Detección sobre múltiples imágenes
# =========================================================
# Wrapper que aplica `detect_faces_in_image` a un dict de imágenes
# etiquetadas y acumula resultados. Es el camino más cómodo cuando se
# está comparando rostros de **varias fotos** distintas (que es el caso
# típico en mendelEmbeddings: una foto del niño + foto del padre + foto
# de la madre).
#
# Depende de: detect_faces_in_image.
def detect_faces_in_images(
    app,
    images: dict[str, np.ndarray],
    pad_x: float = 0.20,
    pad_y: float = 0.35,
):
    """
    Aplica detección a múltiples imágenes etiquetadas.

    Parámetros:
        app: instancia de FaceAnalysis.
        images: dict {photo_label: img_rgb}.
        pad_x, pad_y: padding proporcional para `expand_bbox`.

    Devuelve:
        (annotated_images, all_face_records)
            annotated_images: dict {photo_label: img anotada}.
            all_face_records: lista plana de todos los records, en orden
                de aparición (por foto y dentro de cada foto, por x1).
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
        # `extend` (no `append`) para que sea una lista plana.
        all_face_records.extend(records)

    return annotated_images, all_face_records


# =========================================================
# 5) Búsqueda de un face_record por face_id
# =========================================================
# Pequeño helper para no repetir el `next(... if ...)` en el notebook.
# Si el `face_id` no existe en la lista, propaga `StopIteration`
# (comportamiento heredado del archivo original; el usuario sabe en
# pantalla qué IDs hay porque los acaba de imprimir).
# No depende de funciones propias del módulo.
def get_face_record(all_face_records: list[dict], face_id: str) -> dict:
    """
    Busca un face_record por face_id.

    Parámetros:
        all_face_records: lista devuelta por `detect_faces_in_images`.
        face_id: ID a buscar (ej. "F1_R2").

    Devuelve:
        El primer record que matchee.

    Lanza:
        StopIteration: si no encuentra ningún record con ese face_id.
    """
    return next(r for r in all_face_records if r["face_id"] == face_id)


# =========================================================
# 6) Alineación canónica de un rostro con 5-puntos + margen extra
# =========================================================
# Alinea una cara a un canvas cuadrado de tamaño `image_size` usando los
# 5 keypoints estándar (ojos, nariz, comisuras de la boca) que devuelve
# InsightFace. El `margin_ratio` agrega un margen extra alrededor del
# rostro alineado: el rostro queda más "metido" en el canvas, con
# contexto alrededor (frente + mentón + perfil), lo cual:
#   - mejora la robustez de MediaPipe Face Mesh (que necesita ver todo
#     el contorno),
#   - hace los crops alineados más útiles visualmente.
#
# El algoritmo es:
#   1) Traslada los keypoints del sistema global de la imagen al sistema
#      local del crop expandido.
#   2) Llama a `face_align.estimate_norm` para obtener la matriz afín que
#      lleva esos 5 puntos a las posiciones canónicas del template.
#   3) Ajusta la matriz para reducir el tamaño aparente del rostro
#      (factor `scale`) y centrarlo (shift).
#   4) Aplica `cv2.warpAffine` con bordes replicados (evita franjas
#      negras si la cara queda cerca del borde del crop).
# No depende de funciones propias del módulo.
def align_face_from_record(
    face_record: dict,
    image_size: int = 224,
    margin_ratio: float = 0.18,
) -> np.ndarray:
    """
    Alinea un rostro usando los 5 keypoints, con margen extra opcional.

    Parámetros:
        face_record: dict producido por `detect_faces_in_image`.
        image_size: lado del canvas cuadrado de salida (px). **Debe ser
            múltiplo de 112 o 128** (restricción de
            `insightface.utils.face_align.estimate_norm`). Valores típicos:
            112, 128, 224, 256.
        margin_ratio: 0..0.5. Fracción del canvas reservada como margen
            alrededor del rostro alineado. 0 = rostro ocupa todo; 0.18 =
            ~18% de margen por lado (default).

    Devuelve:
        ndarray uint8 (image_size, image_size, 3) en RGB.

    Lanza:
        ValueError: si no hay keypoints o si `margin_ratio` está fuera de
            rango, o si `estimate_norm` devuelve una matriz inesperada.
    """
    crop_rgb = face_record["crop_rgb"]
    kps = face_record["kps"]

    # Sin keypoints no podemos alinear: error temprano y explícito.
    if kps is None:
        raise ValueError(
            f"El rostro {face_record['face_id']} no tiene keypoints disponibles."
        )

    # Sanidad de margin_ratio: 0.5 colapsaría el rostro a 0 px.
    if not (0.0 <= margin_ratio < 0.5):
        raise ValueError("margin_ratio debe estar entre 0.0 y < 0.5")

    # Traslación de keypoints al sistema local del crop.
    # `bbox` está en coordenadas globales de la imagen; el crop empieza en (x1, y1).
    x1, y1, _, _ = face_record["bbox"]
    kps_local = kps.copy().astype(np.float32)
    kps_local[:, 0] -= x1
    kps_local[:, 1] -= y1

    # `estimate_norm` calcula la transformación afín 2x3 que lleva los
    # 5 keypoints a sus posiciones canónicas en el template de tamaño
    # `image_size`. Algunas versiones de InsightFace devuelven tupla
    # (M, score), otras solo M — manejamos ambos casos.
    est = face_align.estimate_norm(kps_local, image_size=image_size)
    M = est[0] if isinstance(est, tuple) else est
    M = np.asarray(M, dtype=np.float32)

    # Validación de la forma de la matriz. warpAffine necesita estrictamente (2, 3).
    if M.shape != (2, 3):
        raise ValueError(
            f"estimate_norm devolvió shape inesperado: {M.shape}. Se esperaba (2, 3)."
        )

    # Reducción de escala para introducir margen.
    # Si margin_ratio = 0.18, queremos que el rostro ocupe 64% del canvas
    # (100% - 2*18%). Para eso multiplicamos la parte lineal de la matriz
    # por 0.64.
    scale = 1.0 - (2.0 * margin_ratio)
    M_adj = M.copy()
    M_adj[:, :2] *= scale

    # Reajuste del término de traslación: shift para recentrar el rostro
    # ya reducido en el canvas (en vez de quedar pegado a la esquina).
    shift = (image_size * (1.0 - scale)) / 2.0
    M_adj[:, 2] = M_adj[:, 2] * scale + shift

    # Warp final. BORDER_REPLICATE evita franjas negras si el rostro
    # queda cerca de los bordes del crop original.
    aligned_rgb = cv2.warpAffine(
        crop_rgb,
        M_adj,
        (image_size, image_size),
        borderMode=cv2.BORDER_REPLICATE,
    )

    return aligned_rgb


# =========================================================
# 7) Construcción del selected_pair (estructura central de comparación)
# =========================================================
# Recibe la lista global de face_records y dos `face_id` (uno por cara
# a comparar) y devuelve el dict `selected_pair` que el resto del pipeline
# espera (landmarks, regions, comparators, viz).
#
# Estructura del dict de salida:
#   {
#     "face_a": dict,              # face_record completo de A
#     "face_b": dict,              # face_record completo de B
#     "aligned_a": ndarray,        # rostro A alineado (image_size, image_size, 3)
#     "aligned_b": ndarray,        # rostro B alineado (mismo shape)
#     "align_size": int,           # parámetro usado (para reproducibilidad)
#     "align_margin_ratio": float, # parámetro usado
#   }
#
# Depende de: get_face_record, align_face_from_record.
def build_selected_pair(
    all_face_records: list[dict],
    face_id_a: str,
    face_id_b: str,
    align_size: int = 224,
    margin_ratio: float = 0.18,
):
    """
    Construye el `selected_pair` para una comparación A vs B.

    Parámetros:
        all_face_records: lista devuelta por `detect_faces_in_images`.
        face_id_a, face_id_b: IDs de los dos rostros a comparar.
        align_size: lado del canvas de alineación (px).
        margin_ratio: margen relativo (ver `align_face_from_record`).

    Devuelve:
        dict `selected_pair` listo para pasarse a los demás módulos.
    """
    # Búsqueda y alineación de cada rostro de forma independiente.
    face_a = get_face_record(all_face_records, face_id_a)
    face_b = get_face_record(all_face_records, face_id_b)

    aligned_a = align_face_from_record(
        face_a, image_size=align_size, margin_ratio=margin_ratio
    )
    aligned_b = align_face_from_record(
        face_b, image_size=align_size, margin_ratio=margin_ratio
    )

    # Guardamos también los parámetros de alineación para que el dict
    # sea reproducible / autodocumentado.
    return {
        "face_a": face_a,
        "face_b": face_b,
        "aligned_a": aligned_a,
        "aligned_b": aligned_b,
        "align_size": align_size,
        "align_margin_ratio": margin_ratio,
    }
