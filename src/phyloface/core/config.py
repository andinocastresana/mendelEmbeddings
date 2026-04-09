# =========================================
# ID: PHYLOFACE_BACKEND_001
# VERSION: v1.0
# =========================================
# Reemplaza COMPLETO estos 4 archivos:
#
# 1) phyloface/core/config.py
# 2) phyloface/core/detector.py
# 3) phyloface/core/cache.py
# 4) phyloface/app/build_face_cache.py
#
# Qué añade:
# - definición explícita de LIBRARY_NAME y MODEL_NAME en cada corrida
# - guardado de library + model en cache
# - estructura de cache:
#   data/cache/faces/<image_name>/<library_name>/<model_name>/<config_id>/
#
# Backend implementado ahora:
# - insightface
#
# Si en el futuro quieres otra librería, añadimos otro backend en detector.py


# -----------------------------------------
# FILE: phyloface/core/config.py
# -----------------------------------------
from pathlib import Path

# -------------------------
# Proyecto
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# -------------------------
# Entrada
# -------------------------
INPUT_DIR = PROJECT_ROOT / "data/input/img/teams"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# -------------------------
# Cache
# -------------------------
CACHE_ROOT = PROJECT_ROOT / "data/cache/faces"

# -------------------------
# Backend de extracción
# -------------------------
#LIBRARY_NAME = "insightface"   # por ahora solo implementado: "insightface"
#MODEL_NAME = "buffalo_l"       # ejemplos: buffalo_l, buffalo_s, antelopev2

LIBRARY_NAME = "insightface"
MODEL_NAME = "antelopev2"

# -------------------------
# Detección / extracción
# -------------------------
CTX_ID = -1
DET_SIZE = (640, 640)
FACE_SIZE = (112, 112)
MAX_FACES = 11

# -------------------------
# Versionado de cache
# -------------------------
CACHE_SCHEMA_VERSION = "v2"
