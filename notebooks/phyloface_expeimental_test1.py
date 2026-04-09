# %%
# =========================================
# ID: PHYLOFACE_NOTEBOOK_001
# VERSION: v1.0
# =========================================
# NOTEBOOK COMPLETO DE PRUEBA
#
# Supuesto:
# - Ya has creado un único módulo .py con:
#   1) el primer set de funciones (carga, detección, alineación,
#      métricas globales, landmarks densos, regiones v2, etc.)
#   2) el último set de funciones (máscaras poligonales, plotting,
#      comparación regional enmascarada, etc.)
#
# En este ejemplo asumiré que el archivo se llama:
#   phyloface_experimental.py
#
# Si le has puesto otro nombre, cambia la línea de import.
#
# =========================================================
# CELDA 1 - IMPORTS
# =========================================================
import sys
from pathlib import Path

sys.path.append(str(Path("src").resolve()))
from phyloface_experimental_functions import (
    # carga / detección
    load_image,
    init_face_app,
    detect_faces_in_images,
    plot_detected_faces,
    build_selected_pair,
    plot_face_triplet,

    # global
    compute_global_metrics,
    print_global_summary,

    # landmarks densos
    init_face_mesh,
    add_dense_landmarks_to_pair,
    plot_face_with_landmarks,

    # regiones rectangulares
    add_regions_v2_to_pair,
    plot_regions_v2,
    compare_regions_v2,
    print_regional_summary,

    # regiones con máscara
    add_regions_v2_masked_to_pair,
    plot_face_regions_overlay,
    plot_regions_v2_masked,
    plot_region_detail,
    compare_regions_v2_masked,
)

# %%

# =========================================================
# CELDA 2 - CONFIGURACIÓN GENERAL
# =========================================================
# Ajusta estas rutas
PROJECT_ROOT = Path.cwd()

img1_path = PROJECT_ROOT / "data/input/img/fraternos_jovenes.jpg"
img2_path = PROJECT_ROOT / "data/input/img/fraternosChacabuco8.jpg"

# Etiquetas de foto
photo_label_1 = "F1"
photo_label_2 = "F2"

# InsightFace
model_name = "buffalo_l"
det_size = (640, 640)
ctx_id = -1
providers = ["CPUExecutionProvider"]

# Expansión de bbox
face_pad_x = 0.20
face_pad_y = 0.35

# Selección manual del par a comparar
# Primero ejecuta detección y mira los IDs disponibles.
face_id_a = "F1_R1"
face_id_b = "F2_R1"

# Alineación
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
        "| score:", f'{rec["det_score"]:.3f}'
    )


# %%
# =========================================================
# CELDA 6 - VISUALIZAR DETECCIÓN
# =========================================================
plot_detected_faces(
    annotated_images=annotated_images,
    all_face_records=all_face_records,
)

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
plot_face_triplet(selected_pair["face_b"], selected_pair["aligned_b"])


# %%

# =========================================================
# CELDA 9 - MÉTRICAS GLOBALES
# =========================================================
selected_pair = compute_global_metrics(app=app, selected_pair=selected_pair)
print_global_summary(selected_pair)

# También puedes inspeccionar el dict completo si quieres:
print("\nGlobal scores:")
for k, v in selected_pair["global_scores"].items():
    print(f"  {k}: {v:.4f}")

print("\nEmbedding QC:")
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

plot_face_with_landmarks(
    selected_pair["aligned_b"],
    selected_pair["landmarks_b"],
    "aligned_b + dense landmarks",
)


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
# Esto sobrescribe selected_pair["regions_v2"] con la versión enriquecida
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

plot_face_regions_overlay(
    image_rgb=selected_pair["aligned_b"],
    regions=selected_pair["regions_v2"]["B"],
    alpha=0.30,
    title="Face B - overlay regiones",
)


# %%

# =========================================================
# CELDA 18 - VISUALIZAR REGIONES EN LOS 3 MODOS
# =========================================================
# A) Recorte rectangular
plot_regions_v2_masked(selected_pair, mode="rect")

# B) Máscara binaria
plot_regions_v2_masked(selected_pair, mode="mask")

# C) Recorte enmascarado
plot_regions_v2_masked(selected_pair, mode="masked")


# %%

# =========================================================
# CELDA 19 - INSPECCIÓN DETALLADA DE UNA REGIÓN
# =========================================================
# Cambia region_name según lo que quieras inspeccionar
plot_region_detail(selected_pair, side="A", region_name="left_eye")
plot_region_detail(selected_pair, side="B", region_name="left_eye")

plot_region_detail(selected_pair, side="A", region_name="mouth")
plot_region_detail(selected_pair, side="B", region_name="mouth")


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
# Si quieres comparar manualmente el resultado de la métrica rectangular
# y la enmascarada en una misma corrida, rehaz primero la parte rectangular,
# guarda una copia, y luego compara con la enmascarada.

# --- recalcular versión rectangular ---
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
# CELDA 23 - NOTAS PRÁCTICAS
# =========================================================
# Qué tocar habitualmente:
#
# - img1_path / img2_path
# - face_pad_x / face_pad_y
# - face_id_a / face_id_b
# - align_size / align_margin_ratio
#
# Qué mirar para validar:
#
# 1) detección correcta
# 2) alineación razonable
# 3) landmarks densos coherentes
# 4) regiones bien recortadas
# 5) diferencia entre scores rectangulares y masked
#
# Recomendación:
# prueba primero con un par muy claro y luego con uno más difícil.
