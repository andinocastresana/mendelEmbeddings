# %%
# =========================================
# ID: PHYLOFACE_NOTEBOOK_002
# VERSION: v2.0
# =========================================
# NOTEBOOK DE PRUEBA — versión post-migración Tarea #1.
#
# Cambios vs v1.0:
# - Imports actualizados al paquete `phyloface` migrado (en lugar del
#   módulo único `src/phyloface_experimental_functions.py`).
# - Cada llamada a una función `plot_*` queda envuelta por un
#   `save_current(name)` que guarda el plot activo como PNG en
#   `data/output/notebook_runs_<TIMESTAMP>/`. Esto permite revisar
#   visualmente la corrida sin necesitar backend interactivo.
# - El backend de matplotlib se setea a 'Agg' al inicio para que el
#   notebook corra también en entornos sin display (CI, servidor).
#
# Flujo del notebook (idéntico a v1.0):
# 1) Carga 2 imágenes
# 2) Inicializa InsightFace
# 3) Detecta rostros en cada imagen
# 4) Visualiza detección y elige par a comparar manualmente
# 5) Construye selected_pair (alineación canónica)
# 6) Métricas globales (embeddings + coseno/euclídea + QC)
# 7) Landmarks densos (MediaPipe Face Mesh)
# 8) Regiones v2 rectangulares + comparación regional simple
# 9) Regiones v2 con máscara poligonal + comparación regional enmascarada
# 10) Inspecciones varias (overlay, detail, modos rect/mask/masked)

# =========================================================
# CELDA 1 - IMPORTS + SETUP DE OUTPUT
# =========================================================
import sys
from datetime import datetime
from pathlib import Path

# Backend no-interactivo (sirve también con display, pero garantiza que
# corra sin él; los plots se guardan a disco con `save_current`).
# IMPORTANTE: setearlo ANTES de importar matplotlib.pyplot.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Hacemos importable el paquete sin instalación (igual que en smoke tests).
PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# --- Imports desde el paquete migrado ---
from phyloface.core.io import load_image
from phyloface.core.pairs import (
    init_face_app,
    detect_faces_in_images,
    build_selected_pair,
)
from phyloface.core.comparator_global import (
    compute_global_metrics,
    print_global_summary,
)
from phyloface.landmarks import (
    init_face_mesh,
    add_dense_landmarks_to_pair,
)
from phyloface.regions import (
    add_regions_v2_to_pair,
    add_regions_v2_masked_to_pair,
)
from phyloface.comparator_regional import (
    compare_regions_v2,
    compare_regions_v2_masked,
    print_regional_summary,
)
from phyloface.viz import (
    plot_detected_faces,
    plot_face_triplet,
    plot_face_with_landmarks,
    plot_regions_v2,
    plot_face_regions_overlay,
    plot_regions_v2_masked,
    plot_region_detail,
)

# --- Carpeta de salida con timestamp ---
# Vive bajo data/output/ (convención del proyecto: outputs van a data/).
# Sufijo `notebook_runs_` para distinguir de outputs del pipeline real.
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / f"notebook_runs_{RUN_TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[INFO] Plots de esta corrida → {OUTPUT_DIR}")


# Helper para guardar el plot activo a disco con un nombre prefijado por
# el número de celda. Se llama DESPUÉS de cada función `plot_*`.
# Truco: las funciones del paquete llaman `plt.show()` internamente, pero
# `show()` no destruye la figura (en backend Agg es no-op, en interactivo
# bloquea hasta cerrar la ventana). La figura activa sigue siendo `plt.gcf()`
# hasta el próximo `plt.figure()` o `plt.close()`.
_save_counter = 0
def save_current(name: str):
    """Guarda la figura activa como PNG en OUTPUT_DIR y la cierra."""
    global _save_counter
    _save_counter += 1
    fname = f"{_save_counter:02d}_{name}.png"
    fpath = OUTPUT_DIR / fname
    plt.gcf().savefig(fpath, bbox_inches="tight", dpi=120)
    plt.close("all")
    print(f"  [plot guardado] {fname}")


# %%
# =========================================================
# CELDA 2 - CONFIGURACIÓN GENERAL
# =========================================================
img1_path = PROJECT_ROOT / "data/input/img/fraternos_jovenes.jpg"
img2_path = PROJECT_ROOT / "data/input/img/fraternosChacabuco8.jpg"

# Etiquetas de foto (se usan como prefijo de face_id)
photo_label_1 = "F1"
photo_label_2 = "F2"

# InsightFace
model_name = "buffalo_l"
det_size = (640, 640)
ctx_id = -1
providers = ["CPUExecutionProvider"]

# Expansión de bbox al recortar caras
face_pad_x = 0.20
face_pad_y = 0.35

# Selección manual del par a comparar (ver IDs imprimidos tras detección)
face_id_a = "F1_R1"
face_id_b = "F2_R1"

# Alineación (image_size debe ser múltiplo de 112 o 128 — restricción de
# face_align.estimate_norm de InsightFace, documentada en core/pairs.py).
align_size = 224
align_margin_ratio = 0.18


# %%
# =========================================================
# CELDA 3 - CARGA DE IMÁGENES
# =========================================================
img1 = load_image(img1_path)
img2 = load_image(img2_path)

print("Imagen 1:", img1.shape, "|", img1_path.name)
print("Imagen 2:", img2.shape, "|", img2_path.name)


# %%
# =========================================================
# CELDA 4 - INICIALIZAR INSIGHTFACE
# =========================================================
# Tarda 3-5 segundos la primera vez (descarga + load de modelos).
app = init_face_app(
    model_name=model_name,
    det_size=det_size,
    ctx_id=ctx_id,
    providers=providers,
)
print("InsightFace inicializado correctamente.")


# %%
# =========================================================
# CELDA 5 - DETECCIÓN DE ROSTROS
# =========================================================
images = {
    photo_label_1: img1,
    photo_label_2: img2,
}

annotated_images, all_face_records = detect_faces_in_images(
    app=app,
    images=images,
    pad_x=face_pad_x,
    pad_y=face_pad_y,
)

print(f"Total de rostros detectados: {len(all_face_records)}")
for rec in all_face_records:
    print(
        rec["face_id"],
        "| foto:", rec["photo_label"],
        "| bbox:", rec["bbox"],
        "| score:", f'{rec["det_score"]:.3f}',
    )


# %%
# =========================================================
# CELDA 6 - VISUALIZAR DETECCIÓN
# =========================================================
plot_detected_faces(
    annotated_images=annotated_images,
    all_face_records=all_face_records,
)
save_current("detected_faces")


# %%
# =========================================================
# CELDA 7 - CONSTRUIR EL PAR SELECCIONADO
# =========================================================
selected_pair = build_selected_pair(
    all_face_records=all_face_records,
    face_id_a=face_id_a,
    face_id_b=face_id_b,
    align_size=align_size,
    margin_ratio=align_margin_ratio,
)

print("Par seleccionado:")
print(" -", selected_pair["face_a"]["face_id"])
print(" -", selected_pair["face_b"]["face_id"])
print("aligned_a:", selected_pair["aligned_a"].shape)
print("aligned_b:", selected_pair["aligned_b"].shape)


# %%
# =========================================================
# CELDA 8 - VISUALIZAR ALINEACIÓN
# =========================================================
plot_face_triplet(selected_pair["face_a"], selected_pair["aligned_a"])
save_current("face_triplet_a")

plot_face_triplet(selected_pair["face_b"], selected_pair["aligned_b"])
save_current("face_triplet_b")


# %%
# =========================================================
# CELDA 9 - MÉTRICAS GLOBALES
# =========================================================
selected_pair = compute_global_metrics(app=app, selected_pair=selected_pair)
print_global_summary(selected_pair)

print("\nGlobal scores (dict completo):")
for k, v in selected_pair["global_scores"].items():
    print(f"  {k}: {v:.4f}")

print("\nEmbedding QC (dict completo):")
for k, v in selected_pair["embedding_qc"].items():
    print(f"  {k}: {v:.4f}")


# %%
# =========================================================
# CELDA 10 - INICIALIZAR MEDIAPIPE FACE MESH
# =========================================================
face_mesh = init_face_mesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)
print("MediaPipe Face Mesh inicializado correctamente.")


# %%
# =========================================================
# CELDA 11 - LANDMARKS DENSOS
# =========================================================
selected_pair = add_dense_landmarks_to_pair(
    face_mesh=face_mesh,
    selected_pair=selected_pair,
)
print("Landmarks A:", selected_pair["landmarks_a"].shape)
print("Landmarks B:", selected_pair["landmarks_b"].shape)


# %%
# =========================================================
# CELDA 12 - VISUALIZAR LANDMARKS DENSOS
# =========================================================
plot_face_with_landmarks(
    selected_pair["aligned_a"],
    selected_pair["landmarks_a"],
    "aligned_a + dense landmarks",
)
save_current("landmarks_a")

plot_face_with_landmarks(
    selected_pair["aligned_b"],
    selected_pair["landmarks_b"],
    "aligned_b + dense landmarks",
)
save_current("landmarks_b")


# %%
# =========================================================
# CELDA 13 - REGIONES V2 RECTANGULARES
# =========================================================
selected_pair = add_regions_v2_to_pair(selected_pair)
print("Regiones v2 disponibles:")
print(list(selected_pair["regions_v2"]["A"].keys()))


# %%
# =========================================================
# CELDA 14 - VISUALIZAR REGIONES V2 RECTANGULARES
# =========================================================
plot_regions_v2(selected_pair)
save_current("regions_v2_rect")


# %%
# =========================================================
# CELDA 15 - COMPARACIÓN REGIONAL SIMPLE (RECTANGULAR)
# =========================================================
selected_pair = compare_regions_v2(
    selected_pair=selected_pair,
    resize_shape=(64, 64),
)
print_regional_summary(selected_pair)


# %%
# =========================================================
# CELDA 16 - REGIONES V2 CON MÁSCARAS
# =========================================================
# Sobreescribe selected_pair["regions_v2"] con la versión enriquecida
# (bbox + máscara + crop rectangular + crop enmascarado).
selected_pair = add_regions_v2_masked_to_pair(selected_pair)
print("Regiones v2 con máscara generadas correctamente.")
print(list(selected_pair["regions_v2"]["A"].keys()))


# %%
# =========================================================
# CELDA 17 - OVERLAY DE REGIONES SOBRE LA CARA COMPLETA
# =========================================================
plot_face_regions_overlay(
    image_rgb=selected_pair["aligned_a"],
    regions=selected_pair["regions_v2"]["A"],
    alpha=0.30,
    title="Face A - overlay regiones",
)
save_current("overlay_a")

plot_face_regions_overlay(
    image_rgb=selected_pair["aligned_b"],
    regions=selected_pair["regions_v2"]["B"],
    alpha=0.30,
    title="Face B - overlay regiones",
)
save_current("overlay_b")


# %%
# =========================================================
# CELDA 18 - VISUALIZAR REGIONES EN LOS 3 MODOS
# =========================================================
# A) Recorte rectangular
plot_regions_v2_masked(selected_pair, mode="rect")
save_current("regions_v2_masked_mode_rect")

# B) Máscara binaria
plot_regions_v2_masked(selected_pair, mode="mask")
save_current("regions_v2_masked_mode_mask")

# C) Recorte enmascarado
plot_regions_v2_masked(selected_pair, mode="masked")
save_current("regions_v2_masked_mode_masked")


# %%
# =========================================================
# CELDA 19 - INSPECCIÓN DETALLADA DE UNA REGIÓN
# =========================================================
plot_region_detail(selected_pair, side="A", region_name="left_eye")
save_current("detail_A_left_eye")
plot_region_detail(selected_pair, side="B", region_name="left_eye")
save_current("detail_B_left_eye")

plot_region_detail(selected_pair, side="A", region_name="mouth")
save_current("detail_A_mouth")
plot_region_detail(selected_pair, side="B", region_name="mouth")
save_current("detail_B_mouth")


# %%
# =========================================================
# CELDA 20 - COMPARACIÓN REGIONAL ENMASCARADA
# =========================================================
selected_pair = compare_regions_v2_masked(
    selected_pair=selected_pair,
    resize_shape=(64, 64),
)

print("=== COMPARACIÓN REGIONAL ENMASCARADA ===")
for region_name, metrics in sorted(selected_pair["regional_scores"].items()):
    val = metrics["gray_cosine_masked"]
    if val != val:  # NaN check
        print(f"{region_name:16s} | gray_cosine_masked = NaN")
    else:
        print(f"{region_name:16s} | gray_cosine_masked = {val:.4f}")


# %%
# =========================================================
# CELDA 21 - INSPECCIÓN FINAL DEL OBJETO selected_pair
# =========================================================
print("Claves principales en selected_pair:")
for k in selected_pair.keys():
    print(" -", k)

print("\nClaves de global_scores:")
for k in selected_pair["global_scores"].keys():
    print(" -", k)

print("\nClaves de embedding_qc:")
for k in selected_pair["embedding_qc"].keys():
    print(" -", k)

print("\nRegiones disponibles:")
for k in selected_pair["regions_v2"]["A"].keys():
    print(" -", k)


# %%
# =========================================================
# CELDA 22 - OPCIONAL: COMPARACIÓN RÁPIDA RECT VS MASKED
# =========================================================
# Recalcula ambos paths sobre el mismo par y los muestra lado a lado.
# Útil para entender visualmente el impacto del enmascarado vs rectangular.

# --- versión rectangular ---
selected_pair_rect = build_selected_pair(
    all_face_records=all_face_records,
    face_id_a=face_id_a,
    face_id_b=face_id_b,
    align_size=align_size,
    margin_ratio=align_margin_ratio,
)
selected_pair_rect = compute_global_metrics(app=app, selected_pair=selected_pair_rect)
selected_pair_rect = add_dense_landmarks_to_pair(face_mesh=face_mesh, selected_pair=selected_pair_rect)
selected_pair_rect = add_regions_v2_to_pair(selected_pair_rect)
selected_pair_rect = compare_regions_v2(selected_pair_rect, resize_shape=(64, 64))

# --- versión masked ---
selected_pair_masked = build_selected_pair(
    all_face_records=all_face_records,
    face_id_a=face_id_a,
    face_id_b=face_id_b,
    align_size=align_size,
    margin_ratio=align_margin_ratio,
)
selected_pair_masked = compute_global_metrics(app=app, selected_pair=selected_pair_masked)
selected_pair_masked = add_dense_landmarks_to_pair(face_mesh=face_mesh, selected_pair=selected_pair_masked)
selected_pair_masked = add_regions_v2_masked_to_pair(selected_pair_masked)
selected_pair_masked = compare_regions_v2_masked(selected_pair_masked, resize_shape=(64, 64))

print("=== COMPARACIÓN RECTANGULAR vs ENMASCARADA ===")
common_regions = sorted(
    set(selected_pair_rect["regional_scores"].keys()).intersection(
        set(selected_pair_masked["regional_scores"].keys())
    )
)
for region_name in common_regions:
    rect_val = selected_pair_rect["regional_scores"][region_name]["gray_cosine"]
    masked_val = selected_pair_masked["regional_scores"][region_name]["gray_cosine_masked"]
    rect_txt = "NaN" if rect_val != rect_val else f"{rect_val:.4f}"
    masked_txt = "NaN" if masked_val != masked_val else f"{masked_val:.4f}"
    print(f"{region_name:16s} | rect = {rect_txt:>8s} | masked = {masked_txt:>8s}")


# %%
# =========================================================
# CELDA 23 - FIN: RESUMEN DE OUTPUTS
# =========================================================
print()
print(f"[FIN] Notebook completado.")
print(f"[FIN] {_save_counter} plots guardados en: {OUTPUT_DIR}")
print(f"[FIN] Para revisar: ls -la {OUTPUT_DIR}")
