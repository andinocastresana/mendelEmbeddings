#!/usr/bin/env python3
# =========================================
# ID: PHYLOFACE_SPIKE_002
# VERSION: v1.0
# =========================================
# Script de SPIKE menor para Track 2a — paridad MediaPipe Face Mesh JS vs Python.
#
# Contexto:
# El spike anterior (PHYLOFACE_SPIKE_001) validó que ONNX Runtime Web reproduce
# bit-idénticos los embeddings de InsightFace. Este spike valida la otra mitad
# del pipeline cliente: que **MediaPipe Tasks for Web** produce los mismos 478
# landmarks faciales que la versión Python de MediaPipe (`phyloface.landmarks`).
#
# Por qué la imagen de entrada es distinta a la del SPIKE_001:
# - SPIKE_001 (ONNX/embedding): imagen 112x112 sin margen (modelo w600k_r50
#   espera tight crop ArcFace-style).
# - SPIKE_002 (MediaPipe/landmarks): imagen 224x224 con margen 0.18 (MediaPipe
#   FaceMesh necesita ver toda la cara con algo de contexto para que la
#   detección y los landmarks sean estables).
# Por eso este spike usa fixture aparte en `client/public/spike_fixtures_mediapipe/`.
#
# Qué produce:
#   client/public/spike_fixtures_mediapipe/
#   ├── aligned_face_224.png           # imagen 224x224 RGB
#   ├── reference_landmarks.json       # 478 landmarks (x, y en píxeles)
#   └── metadata.json                  # configuración + criterio de éxito
#
# Criterio de éxito (para el cliente JS):
#   - mean_distance_per_landmark < 2.0 px (en imagen 224x224, ~0.9% error)
#   - max_distance_per_landmark  < 5.0 px (tolera outliers chicos)
#   - landmark_count             == reference_count (478 con refine_landmarks)

# -----------------------------------------
# FILE: scripts/verify_mediapipe_web_parity.py
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
from phyloface.landmarks import (  # noqa: E402
    init_face_mesh,
    get_face_mesh_landmarks,
)


def main():
    parser = argparse.ArgumentParser(
        description="Genera fixture de landmarks de MediaPipe para verificar paridad JS vs Python."
    )
    parser.add_argument(
        "--image",
        default="data/input/img/BrunoFondoBlanco.jpeg",
        help="Imagen origen (relativa al root del proyecto).",
    )
    parser.add_argument(
        "--output-dir",
        default="client/public/spike_fixtures_mediapipe",
        help="Directorio donde guardar el fixture (relativo al root).",
    )
    args = parser.parse_args()

    image_path = PROJECT_ROOT / args.image
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Imagen: {image_path}")
    print(f"      Output: {output_dir}\n")

    # === 1. Cargar imagen ===
    img_rgb = load_image(image_path)
    print(f"[2/5] Imagen cargada: {img_rgb.shape}")

    # === 2. InsightFace + detección + alineación ===
    # Igual que el spike anterior, usamos el flujo del motor para que la
    # imagen de entrada al MediaPipe spike sea reproducible. Diferencia clave
    # con SPIKE_001: alineamos a 224x224 con margen 0.18 (NO a 112x112 sin
    # margen). MediaPipe necesita ver toda la cara con contexto.
    print("[3/5] Inicializando InsightFace + detección + alineación 224x224...")
    app = init_face_app(providers=["CPUExecutionProvider"])
    _, face_records = detect_faces_in_image(
        app=app, img_rgb=img_rgb, photo_label="spike_mp",
    )
    if len(face_records) == 0:
        print(f"[!] No se detectaron rostros en {image_path}")
        sys.exit(1)

    face = face_records[0]
    aligned_rgb = align_face_from_record(
        face_record=face, image_size=224, margin_ratio=0.18,
    )
    print(f"      Cara alineada: {aligned_rgb.shape}")

    # === 3. MediaPipe Face Mesh — los 478 landmarks de referencia ===
    print("[4/5] Inicializando MediaPipe Face Mesh + extraendo landmarks...")
    face_mesh = init_face_mesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,  # 478 puntos (vs 468 sin refine, incluye iris)
        min_detection_confidence=0.5,
    )
    landmarks = get_face_mesh_landmarks(face_mesh, aligned_rgb)
    print(
        f"      Landmarks shape: {landmarks.shape}, dtype: {landmarks.dtype}\n"
        f"      Rango X: [{landmarks[:,0].min():.2f}, {landmarks[:,0].max():.2f}]\n"
        f"      Rango Y: [{landmarks[:,1].min():.2f}, {landmarks[:,1].max():.2f}]"
    )

    if landmarks.shape != (478, 2):
        print(f"[!] WARNING: esperaba (478, 2), recibí {landmarks.shape}. ")
        print(f"    Si shape es (468, 2), `refine_landmarks=False`; verificar.")

    # === 4. Guardar fixture ===
    png_path = output_dir / "aligned_face_224.png"
    landmarks_path = output_dir / "reference_landmarks.json"
    meta_path = output_dir / "metadata.json"

    cv2.imwrite(str(png_path), cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR))

    landmarks_payload = {
        "shape": list(landmarks.shape),
        "dtype": "float32",
        "coordinate_system": "image_pixels (x in [0, W], y in [0, H])",
        "image_size_wh": [aligned_rgb.shape[1], aligned_rgb.shape[0]],
        "values": landmarks.tolist(),
    }
    with open(landmarks_path, "w") as f:
        json.dump(landmarks_payload, f)

    metadata = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "source_image_path": str(image_path.relative_to(PROJECT_ROOT)),
        "source_image_shape": list(img_rgb.shape),
        "alignment": {
            "image_size": 224,
            "margin_ratio": 0.18,
            "method": "face_align.estimate_norm (InsightFace) via phyloface.core.pairs",
        },
        "mediapipe": {
            "model": "face_landmarker (mediapipe.solutions.face_mesh)",
            "static_image_mode": True,
            "max_num_faces": 1,
            "refine_landmarks": True,
            "min_detection_confidence": 0.5,
            "n_landmarks": int(landmarks.shape[0]),
        },
        "success_criteria_for_js_client": {
            "mean_distance_per_landmark_max_px": 2.0,
            "max_distance_per_landmark_max_px": 5.0,
            "landmark_count_must_equal_reference": True,
            "note": (
                "MediaPipe Tasks for Web puede tener variaciones sub-pixel "
                "respecto a la versión Python por diferencias en cuantización "
                "del modelo o ordering de operadores. Toleramos hasta 2px de "
                "media + 5px de max. Si supera, investigar."
            ),
        },
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[5/5] Fixture generado en {output_dir}/:")
    for p in (png_path, landmarks_path, meta_path):
        sz = p.stat().st_size
        print(f"        - {p.name:30s}  ({sz:>10,} bytes)")
    print()
    print("[OK] Fixture listo. Próximo paso: spike JS — cargar MediaPipe Tasks "
          "for Web en cliente, correr sobre `aligned_face_224.png`, comparar "
          "landmarks con `reference_landmarks.json`.")


if __name__ == "__main__":
    main()
