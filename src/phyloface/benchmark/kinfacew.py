# =========================================
# ID: PHYLOFACE_BENCHMARK_KINFACEW
# VERSION: v1.0
# =========================================
# Loader del dataset KinFaceW-I/II para calibración (Tarea #6) y protocolo #17.
# Lee TODO desde el .zip (no descomprime a disco) y parsea los splits oficiales
# de `meta_data/*_pairs.mat` (estructura: fold | kin/non-kin | image1 | image2),
# que garantizan reproducibilidad y evitan leakage family-aware (ver pitfalls en
# _meta/BIBLIOGRAFIA_KINSHIP_DATASETS.md).
#
# Decisión: tratamos las caras 64x64 del dataset como YA alineadas y las
# llevamos a 112x112 RGB (validado por el spike: AUC ~0.74 en KinFaceW-I). No
# corremos detección Face Mesh sobre 64px.
#
# Mapeo relación → carpeta/.mat (del ReadMe de KinFaceW): la carpeta father-dau
# contiene archivos fd_*, etc.

import io
import zipfile
from dataclasses import dataclass

import cv2
import numpy as np

INPUT_SIZE = 112

# code -> (mat file, carpeta de imágenes, etiqueta legible, es_cross_genero)
RELATIONS = {
    "fs": ("fs_pairs.mat", "father-son", "Father-Son", False),
    "md": ("md_pairs.mat", "mother-dau", "Mother-Daughter", False),
    "fd": ("fd_pairs.mat", "father-dau", "Father-Daughter", True),
    "ms": ("ms_pairs.mat", "mother-son", "Mother-Son", True),
}


@dataclass(frozen=True)
class Pair:
    fold: int           # 1..5 (split oficial)
    label: int          # 1 = kin, 0 = non-kin
    name1: str          # basename imagen 1 (padre/madre)
    name2: str          # basename imagen 2 (hijo/a)


def _scalar(x):
    a = np.asarray(x).flatten()
    return a[0] if a.size else x


def _to_int(x) -> int:
    return int(_scalar(x))


def _to_str(x) -> str:
    s = _scalar(x)
    if isinstance(s, bytes):
        return s.decode().strip()
    return str(s).strip()


def dataset_root(dataset: str) -> str:
    """Prefijo interno del zip, ej. 'KinFaceW-I/'."""
    return f"{dataset}/"


def load_pairs(zf: zipfile.ZipFile, dataset: str, relation: str) -> list[Pair]:
    """
    Parsea meta_data/<rel>_pairs.mat. Devuelve lista de Pair con fold/label/
    nombres. Requiere scipy (loadmat). Lanza con mensaje claro si la estructura
    no matchea lo esperado (para diagnosticar variantes del .mat).
    """
    from scipy.io import loadmat  # import local: dep que puede no estar

    mat_file = RELATIONS[relation][0]
    mat_path = f"{dataset_root(dataset)}meta_data/{mat_file}"
    raw = zf.read(mat_path)
    md = loadmat(io.BytesIO(raw))

    # La variable de datos es la única clave que no es metadata de loadmat.
    data_keys = [k for k in md.keys() if not k.startswith("__")]
    if not data_keys:
        raise ValueError(f"{mat_path}: sin variables de datos en el .mat")
    var = md[data_keys[0]]
    arr = np.asarray(var, dtype=object)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(
            f"{mat_path}: estructura inesperada shape={arr.shape} "
            f"(esperaba (N,4): fold|kin|img1|img2). keys={data_keys}"
        )

    pairs = []
    for row in arr:
        pairs.append(Pair(
            fold=_to_int(row[0]),
            label=_to_int(row[1]),
            name1=_to_str(row[2]),
            name2=_to_str(row[3]),
        ))
    return pairs


def decode_aligned_rgb(zf: zipfile.ZipFile, dataset: str, relation: str, name: str) -> np.ndarray:
    """Lee una imagen del zip → cara 'pre-alineada' 112x112 RGB."""
    folder = RELATIONS[relation][1]
    path = f"{dataset_root(dataset)}images/{folder}/{name}"
    buf = np.frombuffer(zf.read(path), dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"no pude decodificar {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
        rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    return rgb


def unique_image_names(pairs: list[Pair]) -> list[str]:
    """Nombres únicos de imagen referenciados por una lista de pares."""
    names = set()
    for p in pairs:
        names.add(p.name1)
        names.add(p.name2)
    return sorted(names)
