#!/usr/bin/env python3
# =========================================
# ID: PHYLOFACE_SPIKE_003
# VERSION: v1.0
# =========================================
# Script de SPIKE para Track 2a — paridad de la alineación canónica JS vs Python.
#
# Contexto en la cadena de spikes del Track 2a:
#   SPIKE_001 (ONNX)       ✅ — embeddings idénticos JS vs Python.
#   SPIKE_002 (MediaPipe)  ✅ — 478 landmarks similares JS vs Python.
#   SPIKE_003 (este)       ⏳ — la pieza intermedia: dado un crop y los 5 kps
#                                de InsightFace, replicar `align_face_from_record`
#                                en TS y obtener la imagen 112x112 que entra al
#                                modelo, pixel-a-pixel idéntica a Python.
#
# Por qué este spike es crítico:
# El pipeline cliente del comparador anónimo es:
#     crop + 5 kps  -->  alineación 112x112  -->  ONNX  -->  embedding
# Sin alineación JS no hay forma de cerrar el loop browser-only. Si esta pieza
# diverge, los embeddings calculados en el cliente NO serán comparables con los
# del motor Python — y todo lo que viene después (UI MVP, integración con
# MediaPipe para detección+landmarks, server Track 2b para refinamiento) se cae.
#
# Decisión de alcance (registrada en sesión 2026-05-21):
# Este spike valida SOLO la alineación, asumiendo que el cliente JS ya tiene los
# 5 kps de InsightFace serializados desde el fixture. La detección JS + el mapeo
# de los 6 kps de MediaPipe Face Detector a los 5 kps que `estimate_norm` espera
# se difiere a un spike siguiente (#004), para aislar variables.
#
# Algoritmo que el JS tiene que replicar (de `align_face_from_record` y de
# `insightface.utils.face_align.estimate_norm`):
#   1) Toma 5 kps en coordenadas locales del crop (ya pre-trasladados con -bbox).
#   2) Construye el template ArcFace destino:
#        arcface_dst = [[38.2946, 51.6963], [73.5318, 51.5014],
#                       [56.0252, 71.7366], [41.5493, 92.3655],
#                       [70.7299, 92.2041]]
#        Si image_size es múltiplo de 112: ratio = image_size/112, diff_x = 0
#        Si image_size es múltiplo de 128: ratio = image_size/128, diff_x = 8*ratio
#        dst = arcface_dst * ratio; dst[:, 0] += diff_x
#   3) Resuelve una `SimilarityTransform` (rotación + escala + traslación sin
#      shear) que mapea los 5 kps al template `dst`. Usa Umeyama (1991) via
#      `skimage.transform.SimilarityTransform.estimate(lmk, dst)`.
#      → Devuelve matriz afín 2x3 = M.
#   4) Para margin_ratio > 0, recalcula M_adj:
#        scale = 1 - 2*margin_ratio
#        M_adj[:, :2] = M[:, :2] * scale
#        M_adj[:, 2]  = M[:, 2] * scale + (image_size * (1-scale))/2
#      Para margin_ratio = 0 (caso del modelo ONNX), M_adj = M.
#   5) Aplica `cv2.warpAffine` con interpolación BILINEAR (default) y
#      borderMode = BORDER_REPLICATE (clamp-to-edge).
#
# Qué produce este script:
#   client/public/spike_fixtures_alignment/
#   ├── crop_rgb.png             # crop expandido del que parte la alineación.
#   │                            # El JS recibe ESTO como entrada (no la
#   │                            # imagen original entera).
#   ├── aligned_face_112.png     # referencia: salida esperada del warp 112x112.
#   ├── landmarks.json           # 5 kps en coords locales del crop +
#   │                            # template arcface_dst + matriz M de referencia.
#   └── metadata.json            # configuración + criterio de éxito.
#
# Criterio de éxito (para el cliente JS):
#   - mean_abs_pixel_diff < 1.0   (sobre uint8 [0..255], imagen 112x112x3)
#   - max_abs_pixel_diff  <= 5    (toleramos algún píxel borde por diferencias
#                                  sub-pixel en la interpolación bilineal)
#   - shape match         (112, 112, 3) uint8
#
# Por qué no exigimos paridad EXACTA (max_diff=0):
# Aun replicando bilineal + clamp-to-edge bit a bit, la aritmética de coma
# flotante puede generar diferencias de 1 en un puñado de píxeles en bordes
# muy oblicuos. La tolerancia chica es por ese ruido numérico, NO por permitir
# desvíos algorítmicos. Si fallan más de unos pocos píxeles, el algoritmo JS
# está mal.

# -----------------------------------------
# FILE: scripts/verify_alignment_web_parity.py
# -----------------------------------------

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phyloface.core.io import load_image  # noqa: E402
from phyloface.core.pairs import (  # noqa: E402
    init_face_app,
    detect_faces_in_image,
    align_face_from_record,
)
from insightface.utils import face_align  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Genera fixture para verificar paridad de la alineación canónica "
            "JS vs Python (Track 2a, spike #003)."
        )
    )
    parser.add_argument(
        "--image",
        default="data/input/img/BrunoFondoBlanco.jpeg",
        help="Imagen origen (relativa al root del proyecto).",
    )
    parser.add_argument(
        "--output-dir",
        default="client/public/spike_fixtures_alignment",
        help="Directorio donde guardar el fixture (relativo al root).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=112,
        help=(
            "Tamaño del canvas alineado. Múltiplo de 112 o 128. "
            "Default 112 (el que entra al modelo ONNX w600k_r50)."
        ),
    )
    parser.add_argument(
        "--margin-ratio",
        type=float,
        default=0.0,
        help=(
            "Margen como fracción del canvas. 0 = cara tight (caso del modelo "
            "ONNX). Default 0.0."
        ),
    )
    args = parser.parse_args()

    image_path = PROJECT_ROOT / args.image
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/7] Imagen:    {image_path}")
    print(f"      Output:    {output_dir}")
    print(f"      Alineación: {args.image_size}x{args.image_size}, "
          f"margin_ratio={args.margin_ratio}\n")

    # === 1. Cargar imagen + detección + obtener face_record ===
    img_rgb = load_image(image_path)
    print(f"[2/7] Imagen cargada: {img_rgb.shape}")

    print("[3/7] Inicializando InsightFace + detección...")
    app = init_face_app(providers=["CPUExecutionProvider"])
    _, face_records = detect_faces_in_image(
        app=app,
        img_rgb=img_rgb,
        photo_label="spike_align",
        pad_x=0.20,
        pad_y=0.35,
    )
    if len(face_records) == 0:
        print(f"[!] No se detectaron rostros en {image_path}")
        sys.exit(1)

    face = face_records[0]
    print(
        f"      face_id={face['face_id']}, score={face['det_score']:.3f}, "
        f"bbox={face['bbox']}"
    )

    # === 2. Calcular kps_local (lo que el JS va a usar como input) ===
    # `align_face_from_record` traslada los kps del sistema global de la imagen
    # al sistema local del crop expandido restando (x1, y1). El JS recibe el
    # crop_rgb y los kps_local, así que tiene que hacer EXACTAMENTE eso —
    # pero la traslación la hacemos nosotros acá para que el JS reciba los
    # kps ya en el sistema correcto. (El JS no tiene la bbox; le servimos los
    # kps "listos para usar".)
    crop_rgb = face["crop_rgb"]
    kps_global = face["kps"]
    x1, y1, _, _ = face["bbox"]
    kps_local = kps_global.copy().astype(np.float32)
    kps_local[:, 0] -= x1
    kps_local[:, 1] -= y1
    print(
        f"[4/7] kps_local (post-traslación a sistema del crop):\n"
        f"      {kps_local.tolist()}"
    )

    # === 3. Generar la imagen alineada de referencia (lo que el JS debe matchear) ===
    aligned_rgb_ref = align_face_from_record(
        face_record=face,
        image_size=args.image_size,
        margin_ratio=args.margin_ratio,
    )
    print(
        f"[5/7] aligned_rgb_ref: shape={aligned_rgb_ref.shape}, "
        f"dtype={aligned_rgb_ref.dtype}"
    )

    # === 4. Calcular y guardar la matriz M (estimate_norm + margin adj) ===
    # Útil para que el cliente JS pueda debuggear en path "easy" (recibir M
    # ya calculada → solo testear warpAffine) vs path "completo" (recibir
    # solo los kps → estimar M en JS + warpAffine). Si solo falla "completo",
    # el bug está en la implementación TS de Umeyama. Si fallan ambos, está
    # en el warpAffine.
    M = face_align.estimate_norm(kps_local, image_size=args.image_size)
    M = np.asarray(M, dtype=np.float32)
    scale = 1.0 - (2.0 * args.margin_ratio)
    M_adj = M.copy()
    M_adj[:, :2] *= scale
    shift = (args.image_size * (1.0 - scale)) / 2.0
    M_adj[:, 2] = M_adj[:, 2] * scale + shift
    print(
        f"      M (estimate_norm sin margen):\n        {M.tolist()}\n"
        f"      M_adj (con margin_ratio={args.margin_ratio}):\n"
        f"        {M_adj.tolist()}"
    )

    # === 5. Sanity check Python: aplicar M_adj manualmente y comparar con la
    # imagen producida por `align_face_from_record` (deberían ser idénticas) ===
    aligned_manual = cv2.warpAffine(
        crop_rgb,
        M_adj,
        (args.image_size, args.image_size),
        borderMode=cv2.BORDER_REPLICATE,
    )
    sanity_diff = np.abs(
        aligned_rgb_ref.astype(np.int16) - aligned_manual.astype(np.int16)
    )
    sanity_max = int(sanity_diff.max())
    sanity_mean = float(sanity_diff.mean())
    print(
        f"[6/7] Sanity check Python (warpAffine manual vs align_face_from_record):\n"
        f"        max_abs_diff  = {sanity_max} (sobre uint8)\n"
        f"        mean_abs_diff = {sanity_mean:.6f}"
    )
    if sanity_max != 0:
        print(
            "[!!] Sanity check FALLÓ: la reconstrucción manual no matchea "
            "`align_face_from_record`. Es un bug del script, no del JS. Frenar."
        )
        sys.exit(2)
    print("      OK: ambas rutas Python dan el mismo resultado bit-a-bit.")

    # === 6. Guardar fixture ===
    crop_png_path = output_dir / "crop_rgb.png"
    aligned_png_path = output_dir / "aligned_face_112.png"
    landmarks_path = output_dir / "landmarks.json"
    meta_path = output_dir / "metadata.json"

    cv2.imwrite(str(crop_png_path), cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(
        str(aligned_png_path),
        cv2.cvtColor(aligned_rgb_ref, cv2.COLOR_RGB2BGR),
    )

    landmarks_payload = {
        "kps_local": {
            "shape": list(kps_local.shape),
            "dtype": "float32",
            "coordinate_system": (
                "image_pixels in the crop_rgb canvas (x in [0, crop_W], "
                "y in [0, crop_H])"
            ),
            "crop_size_wh": [crop_rgb.shape[1], crop_rgb.shape[0]],
            "values": kps_local.tolist(),
            "order": "left_eye, right_eye, nose, left_mouth, right_mouth",
        },
        "arcface_dst_template": {
            "image_size_reference": 112,
            "values": face_align.arcface_dst.tolist(),
            "note": (
                "Template canónico ArcFace en sistema 112x112. Para image_size "
                "distinto, escalar como ratio = image_size/112 si es múltiplo "
                "de 112, o ratio = image_size/128 + diff_x = 8*ratio si es "
                "múltiplo de 128. Ver `insightface.utils.face_align.estimate_norm`."
            ),
        },
        "reference_matrix_M": {
            "shape": list(M.shape),
            "dtype": "float32",
            "values": M.tolist(),
            "note": (
                "Matriz afín 2x3 que devuelve estimate_norm para los kps_local "
                "y este image_size, SIN ajuste de margen. Sirve para que el "
                "cliente JS testee solo el warpAffine (path easy) usando esta "
                "M en vez de calcular Umeyama."
            ),
        },
        "reference_matrix_M_adj": {
            "shape": list(M_adj.shape),
            "dtype": "float32",
            "values": M_adj.tolist(),
            "note": (
                "Matriz afín 2x3 final = M ajustada por margin_ratio. Es la M "
                "que se pasa a warpAffine para producir aligned_face_112.png. "
                "Con margin_ratio=0, M_adj == M."
            ),
        },
    }
    with open(landmarks_path, "w") as f:
        json.dump(landmarks_payload, f, indent=2)

    metadata = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "source_image_path": str(image_path.relative_to(PROJECT_ROOT)),
        "source_image_shape": list(img_rgb.shape),
        "crop": {
            "shape": list(crop_rgb.shape),
            "note": (
                "El crop_rgb es la imagen expandida del face_record (recorte "
                "con pad_x=0.20, pad_y=0.35 alrededor de la bbox detectada). "
                "Es el input directo del JS al pipeline de alineación."
            ),
        },
        "alignment": {
            "image_size": args.image_size,
            "margin_ratio": args.margin_ratio,
            "method": "face_align.estimate_norm + warpAffine (BORDER_REPLICATE)",
            "interpolation": "bilinear (cv2 default INTER_LINEAR)",
            "border_mode": "BORDER_REPLICATE (clamp-to-edge)",
            "py_function": "phyloface.core.pairs.align_face_from_record",
        },
        "sanity_check_python": {
            "warpAffine_manual_vs_align_face_from_record": {
                "max_abs_pixel_diff": sanity_max,
                "mean_abs_pixel_diff": sanity_mean,
                "passed": sanity_max == 0,
            },
        },
        "success_criteria_for_js_client": {
            "mean_abs_pixel_diff_max": 1.0,
            "max_abs_pixel_diff_max": 5,
            "shape_must_match": [args.image_size, args.image_size, 3],
            "dtype_must_match": "uint8",
            "note": (
                "Tolerancia chica por ruido numérico de la aritmética de "
                "coma flotante en la interpolación bilineal de bordes muy "
                "oblicuos. Si supera, el algoritmo JS está mal."
            ),
        },
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[7/7] Fixture generado en {output_dir}/:")
    for p in (crop_png_path, aligned_png_path, landmarks_path, meta_path):
        sz = p.stat().st_size
        print(f"        - {p.name:30s}  ({sz:>10,} bytes)")
    print()
    print(
        "[OK] Fixture listo. Próximo paso: SpikeAlignment.tsx en el cliente "
        "(implementar Umeyama + warpAffine bilineal + BORDER_REPLICATE)."
    )


if __name__ == "__main__":
    main()
