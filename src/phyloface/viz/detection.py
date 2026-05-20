# =========================================
# ID: PHYLOFACE_VIZ_001
# VERSION: v1.0
# =========================================
# Origen de las funciones (migración Tarea #1, Paso 7):
#   src/phyloface_experimental_functions.py
#   - plot_detected_faces  (línea 208)
#   - plot_face_triplet    (línea 340)
#
# Esta entrega cubre 3 archivos (mismo ID):
#   1) phyloface/viz/detection.py   (este archivo)
#   2) phyloface/viz/landmarks.py
#   3) phyloface/viz/regions.py
#
# Qué hace este módulo:
# - Visualizaciones del **bloque de detección + alineación** (módulo
#   compañero: `phyloface.core.pairs`).
# - Dos funciones:
#     plot_detected_faces : panel multi-imagen con las bboxes detectadas
#                           anotadas + face_id sobre cada rostro. Útil
#                           para inspección manual antes de elegir el par.
#     plot_face_triplet   : panel 1x3 para un único rostro mostrando
#                           crop, crop+keypoints y resultado de alineación.
#                           Útil para validar visualmente la alineación
#                           antes de calcular embeddings.
#
# Decisiones de diseño:
# - Todas las funciones llaman `plt.show()` al final. En backend
#   interactivo (notebook, Qt, TkAgg) abren ventana; en backend no
#   interactivo (Agg, usado en CI y tests) son no-op gráfica pero
#   ejecutan el rendering interno (sirve para validar la composición).
# - No se devuelve nada — son funciones de "lado-efecto visual".
#   Si en el futuro alguien necesita el `fig` para guardar a disco,
#   habrá que agregar un parámetro `save_path` o devolver el fig.
#   Hoy se mantiene la API tal cual el archivo original.

# -----------------------------------------
# FILE: phyloface/viz/detection.py
# -----------------------------------------

import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1) Panel multi-imagen con bboxes + face_id anotado
# =========================================================
# Muestra una imagen anotada por foto (ya viene con bboxes dibujadas por
# `detect_faces_in_image`) y agrega un label amarillo con el face_id
# sobre cada rostro. Es la herramienta primaria para que el usuario elija
# manualmente qué dos rostros comparar.
# No depende de funciones propias del módulo.
def plot_detected_faces(
    annotated_images: dict,
    all_face_records: list,
):
    """
    Visualiza imágenes anotadas con IDs de rostro encima.

    Parámetros:
        annotated_images: dict {photo_label: img_rgb_con_bboxes}.
        all_face_records: lista plana de face_records (típicamente la
            que devuelve `detect_faces_in_images`).
    """
    # Una columna por foto. Si solo hay una foto, matplotlib devuelve
    # un Axes individual (no array) — lo wrappeamos en lista para
    # poder iterar uniforme.
    fig, axes = plt.subplots(1, len(annotated_images), figsize=(8 * len(annotated_images), 8))
    if len(annotated_images) == 1:
        axes = [axes]

    for ax, (photo_label, img_rgb) in zip(axes, annotated_images.items()):
        ax.imshow(img_rgb)
        ax.set_title(f"{photo_label} - rostros detectados")
        ax.axis("off")

        # Para cada record que pertenezca a esta foto, dibujamos el face_id
        # sobre la bbox (esquina superior izquierda, ligeramente arriba).
        for rec in all_face_records:
            if rec["photo_label"] == photo_label:
                x1, y1, _, _ = rec["bbox"]
                ax.text(
                    x1,
                    # max(10, y1-10): evita que el label se salga por arriba
                    # si la cara está pegada al borde superior.
                    max(10, y1 - 10),
                    rec["face_id"],
                    fontsize=12,
                    bbox=dict(facecolor="yellow", alpha=0.7, edgecolor="black"),
                )

    plt.tight_layout()
    plt.show()


# =========================================================
# 2) Panel 1x3 para un rostro: crop / crop+keypoints / alineado
# =========================================================
# Diagnóstico rápido del pipeline de alineación para un único rostro:
#   panel 1: el recorte tal como sale del detector (con bbox expandida).
#   panel 2: el mismo recorte con los 5 keypoints superpuestos.
#   panel 3: el resultado tras `align_face_from_record`.
# Es la herramienta para detectar visualmente si la alineación quedó torcida,
# si los keypoints están mal ubicados, o si el padding fue insuficiente.
# No depende de funciones propias del módulo.
def plot_face_triplet(face_record: dict, aligned_rgb: np.ndarray):
    """
    Muestra recorte / recorte+keypoints / alineado para un face_record.

    Parámetros:
        face_record: dict producido por `detect_faces_in_image`. Debe
            tener 'crop_rgb', 'kps', 'bbox' y 'face_id'.
        aligned_rgb: imagen alineada (typically devuelta por
            `align_face_from_record`).
    """
    crop_rgb = face_record["crop_rgb"]
    kps = face_record["kps"]

    # Los `kps` vienen en coordenadas globales de la imagen original;
    # para superponer sobre el `crop_rgb` hay que trasladarlos al
    # sistema local del crop (restando la esquina (x1, y1) de la bbox).
    x1, y1, _, _ = face_record["bbox"]
    kps_local = kps.copy().astype(np.float32)
    kps_local[:, 0] -= x1
    kps_local[:, 1] -= y1

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))

    # Panel 0: solo el crop.
    axes[0].imshow(crop_rgb)
    axes[0].set_title(f"{face_record['face_id']}\nrecorte")
    axes[0].axis("off")

    # Panel 1: crop + scatter de los 5 keypoints en rojo (color default).
    axes[1].imshow(crop_rgb)
    axes[1].scatter(kps_local[:, 0], kps_local[:, 1], s=40)
    axes[1].set_title(f"{face_record['face_id']}\nkeypoints")
    axes[1].axis("off")

    # Panel 2: el rostro ya alineado por nuestro pipeline.
    axes[2].imshow(aligned_rgb)
    axes[2].set_title(f"{face_record['face_id']}\nalineado")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()
