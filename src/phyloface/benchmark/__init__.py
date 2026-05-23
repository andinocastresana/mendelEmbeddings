# =========================================
# ID: PHYLOFACE_BENCHMARK_INIT
# VERSION: v1.0
# =========================================
# Subpaquete de benchmarking / calibración (área "B" del proyecto).
# - kinfacew:    loader del dataset KinFaceW (parsea meta_data/*_pairs.mat).
# - calibration: lógica pura de calibración de umbrales (5-fold CV, ROC/AUC).
#
# Iniciado en Tarea #6 (calibración de umbrales). Reutilizable por #17/#18
# (protocolos KinFaceW / TSKinFace).

from . import calibration  # noqa: F401
from . import kinfacew  # noqa: F401
