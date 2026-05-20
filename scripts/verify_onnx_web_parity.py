#!/usr/bin/env python3
# =========================================
# ID: PHYLOFACE_SPIKE_001
# VERSION: v1.0
# =========================================
# Script de SPIKE para Track 2a — paridad ONNX Runtime Web vs Python.
#
# Objetivo:
# Producir un "fixture" reproducible que el cliente JS (sub-paso D/E del spike)
# va a consumir para verificar que `onnxruntime-web` carga w600k_r50.onnx y
# produce embeddings cuasi bit-idénticos a la implementación Python.
#
# Por qué este spike es crítico:
# Si la inferencia ONNX en browser diverge significativamente de Python, el
# plan híbrido "cliente-pesado / server-liviano" v0.1 se cae. La validación
# acá es el guardrail antes de invertir tiempo en armar el resto del cliente.
# Ver `_meta/arquitectura_web/v0.1_2026-05-20_arquitectura_web.md` §9 y §10.
#
# Qué produce este script:
#   client/public/spike_fixtures/
#   ├── aligned_face.png             # imagen 112x112 RGB uint8 (entrada al
#   │                                # modelo en formato visual; el JS la
#   │                                # carga, normaliza y pasa al modelo).
#   ├── aligned_face_tensor.json     # tensor (1, 3, 112, 112) ya normalizado
#   │                                # con (px - 127.5) / 128.0, en orden CHW.
#   │                                # Permite aislar si los desvíos vienen
#   │                                # del preprocessing JS o del modelo en sí.
#   ├── reference_embedding.json     # embedding de referencia 512-D float32
#   │                                # calculado con el motor Python.
#   └── metadata.json                # info sobre la corrida (imagen origen,
#                                    # modelo, normalización aplicada,
#                                    # criterio de éxito, timestamp).
#
# Cómo verificar paridad desde el cliente JS:
#   - Path "easy": cargar aligned_face_tensor.json -> pasar tensor al modelo
#     -> comparar con reference_embedding.json. Si falla, problema en el modelo
#     JS o sus operadores en backend WebGPU/WebGL/WASM.
#   - Path "completo": cargar aligned_face.png -> aplicar normalización
#     idéntica en JS -> pasar al modelo -> comparar. Si el path easy pasa
#     y este falla, problema en preprocessing JS (BGR/RGB, HWC/CHW, fórmula).
#
# Criterio de éxito (documentado en v0.1 §10):
#   cosine_similarity(emb_py, emb_js) > 0.9999
#   max(|emb_py - emb_js|) < 1e-3

# -----------------------------------------
# FILE: scripts/verify_onnx_web_parity.py
# -----------------------------------------

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


# Carpeta src/ al path para importar phyloface sin instalar el paquete.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phyloface.core.io import load_image  # noqa: E402
from phyloface.core.pairs import (  # noqa: E402
    init_face_app,
    detect_faces_in_image,
    align_face_from_record,
)
from phyloface.core.embedder import (  # noqa: E402
    get_recognition_model,
    extract_embedding_from_aligned,
)
from phyloface.core.metrics import cosine_similarity  # noqa: E402


# =========================================================
# Preprocessing que el cliente JS va a tener que replicar
# =========================================================
# Fuente: InsightFace usa internamente una normalización del tipo
#   (px - input_mean) / input_std
# donde `input_mean` e `input_std` están **embedidos en el modelo
# concreto** y se exponen como atributos del submodelo cargado:
#   rec_model.input_mean, rec_model.input_std
#
# Observación CRÍTICA capturada durante este spike (sin sanity check
# habríamos perdido horas debuggeando el cliente JS):
# El bundle "buffalo_l", submodelo recognition (w600k_r50.onnx), usa
# mean=127.5 y std=127.5 — NO los valores "estándar de ArcFace"
# (mean=127.5, std=128.0). La diferencia chica numéricamente (0.5/128
# = 0.4%) acumulada × 50k píxeles × las activaciones del modelo
# diverge el embedding final (similitud cae de ~1.0 a ~0.95).
#
# Por eso este script lee mean/std del modelo en runtime y los guarda
# en metadata.json para que el cliente JS use exactamente los mismos.
#
# Fórmula:
#   1) Asume entrada RGB uint8 (H, W, 3), con H=W=112.
#   2) Cast a float32.
#   3) Normaliza: (px - mean) / std  (mean/std del modelo).
#   4) HWC → CHW (transpose ejes).
#   5) Batch dim → (1, 3, 112, 112).
def preprocess_for_recognition(
    aligned_rgb: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    """Replica el preprocessing interno del recognition model de InsightFace."""
    assert aligned_rgb.shape == (112, 112, 3), (
        f"Esperaba shape (112, 112, 3), recibí {aligned_rgb.shape}"
    )
    assert aligned_rgb.dtype == np.uint8, (
        f"Esperaba uint8, recibí {aligned_rgb.dtype}"
    )

    # Float + normalize con los parámetros REALES del modelo.
    x = aligned_rgb.astype(np.float32)
    x = (x - mean) / std

    # HWC -> CHW.
    x = x.transpose(2, 0, 1)

    # Batch dim.
    x = np.expand_dims(x, axis=0)
    return x.astype(np.float32)


# =========================================================
# Inferencia "manual" llamando directamente al ONNX runtime
# =========================================================
# Pasamos el tensor preprocesado al modelo via la session ONNX. Si nuestro
# preprocessing manual está correcto, debería matchear get_feat(aligned_rgb)
# dentro del orden de epsilon de punto flotante.
def run_model_manual(rec_model, tensor_preprocessed: np.ndarray) -> np.ndarray:
    """Corre la session ONNX manualmente y devuelve embedding raw."""
    session = rec_model.session
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    outputs = session.run([output_name], {input_name: tensor_preprocessed})
    return outputs[0].flatten().astype(np.float32)


# =========================================================
# Pipeline principal
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Genera fixture para verificar paridad ONNX Web vs Python."
    )
    parser.add_argument(
        "--image",
        default="data/input/img/BrunoFondoBlanco.jpeg",
        help="Imagen origen (relativa al root del proyecto).",
    )
    parser.add_argument(
        "--output-dir",
        default="client/public/spike_fixtures",
        help="Directorio donde guardar el fixture (relativo al root).",
    )
    parser.add_argument(
        "--model-name",
        default="buffalo_l",
        help="Bundle de modelos InsightFace (default: buffalo_l).",
    )
    args = parser.parse_args()

    image_path = PROJECT_ROOT / args.image
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/7] Imagen: {image_path}")
    print(f"      Output: {output_dir}\n")

    # === 1. Cargar imagen ===
    img_rgb = load_image(image_path)
    print(f"[2/7] Imagen cargada: {img_rgb.shape}, dtype={img_rgb.dtype}")

    # === 2. InsightFace ===
    print("[3/7] Inicializando InsightFace (puede tardar 3-5s)...")
    app = init_face_app(model_name=args.model_name, providers=["CPUExecutionProvider"])
    print("      OK.")

    # === 3. Detección ===
    _, face_records = detect_faces_in_image(
        app=app,
        img_rgb=img_rgb,
        photo_label="spike",
        pad_x=0.20,
        pad_y=0.35,
    )
    if len(face_records) == 0:
        print(f"[!] No se detectaron rostros en {image_path}")
        sys.exit(1)

    face = face_records[0]
    print(
        f"[4/7] Detectada cara: face_id={face['face_id']}, "
        f"score={face['det_score']:.3f}, bbox={face['bbox']}"
    )

    # === 4. Alineación a 112x112 sin margen extra ===
    # El modelo w600k_r50 espera entradas alineadas a 112x112 con la cara
    # ocupando casi todo el canvas (sin contexto extra). margin_ratio=0
    # → cara "tight". image_size=112 múltiplo de 112, cumple la restricción
    # de face_align.estimate_norm.
    aligned_rgb = align_face_from_record(
        face_record=face,
        image_size=112,
        margin_ratio=0.0,
    )
    print(f"[5/7] Cara alineada: {aligned_rgb.shape}, dtype={aligned_rgb.dtype}")

    # === 5. Embedding "de referencia" via get_feat (camino que usa el motor) ===
    rec_model = get_recognition_model(app)
    emb_ref = extract_embedding_from_aligned(rec_model, aligned_rgb)
    print(
        f"[6/7] Embedding (get_feat): shape={emb_ref.shape}, "
        f"dtype={emb_ref.dtype}, norm={np.linalg.norm(emb_ref):.4f}"
    )

    # === 6. Verificación: nuestro preprocessing manual debería matchear get_feat ===
    # Si esto falla, el preprocessing JS también va a fallar — significa que
    # nuestra interpretación de "qué hace get_feat" está equivocada.
    # mean/std vienen DEL MODELO (no se asume el "estándar ArcFace" 127.5/128).
    rec_mean = float(rec_model.input_mean)
    rec_std = float(rec_model.input_std)
    print(f"      Modelo: input_mean={rec_mean}, input_std={rec_std}")
    tensor = preprocess_for_recognition(aligned_rgb, mean=rec_mean, std=rec_std)
    emb_manual = run_model_manual(rec_model, tensor)
    sim_manual_vs_ref = cosine_similarity(emb_ref, emb_manual)
    max_abs_diff = float(np.max(np.abs(emb_ref - emb_manual)))
    print(
        f"      Sanity check (manual preprocessing vs get_feat):\n"
        f"        cosine_similarity = {sim_manual_vs_ref:.6f}\n"
        f"        max |diff|        = {max_abs_diff:.6e}"
    )
    if sim_manual_vs_ref < 0.9999:
        print(
            f"[!!] WARNING: el preprocessing manual NO matchea get_feat. "
            f"Eso significa que InsightFace hace algo distinto a lo que asumimos "
            f"(normalización, BGR/RGB, etc.). El spike JS va a fallar."
        )
    else:
        print("      OK: nuestro preprocessing manual reproduce get_feat.")

    # === 7. Guardar fixture ===
    png_path = output_dir / "aligned_face.png"
    tensor_path = output_dir / "aligned_face_tensor.json"
    emb_path = output_dir / "reference_embedding.json"
    meta_path = output_dir / "metadata.json"

    # PNG: cv2 espera BGR, hacemos la conversión inversa.
    cv2.imwrite(str(png_path), cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR))

    # Tensor pre-procesado (CHW float32, ya normalizado). Lo aplanamos +
    # guardamos shape para que el JS pueda reconstruirlo.
    tensor_payload = {
        "shape": list(tensor.shape),
        "dtype": "float32",
        "layout": "NCHW",
        "normalization": f"(px_uint8 - {rec_mean}) / {rec_std}",
        "input_mean": rec_mean,
        "input_std": rec_std,
        "data_flat": tensor.flatten().tolist(),
    }
    with open(tensor_path, "w") as f:
        json.dump(tensor_payload, f)

    # Embedding de referencia.
    emb_payload = {
        "shape": list(emb_ref.shape),
        "dtype": "float32",
        "norm_l2": float(np.linalg.norm(emb_ref)),
        "values": emb_ref.tolist(),
    }
    with open(emb_path, "w") as f:
        json.dump(emb_payload, f)

    # Metadata para auditoría/debug desde el cliente.
    metadata = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "source_image_path": str(image_path.relative_to(PROJECT_ROOT)),
        "source_image_shape": list(img_rgb.shape),
        "model": {
            "library": "insightface",
            "bundle": args.model_name,
            "recognition_submodel": "w600k_r50.onnx",
        },
        "alignment": {
            "image_size": 112,
            "margin_ratio": 0.0,
            "method": "face_align.estimate_norm (InsightFace)",
        },
        "preprocessing": {
            "color": "RGB uint8 -> float32",
            "normalize": f"(px - {rec_mean}) / {rec_std}",
            "input_mean": rec_mean,
            "input_std": rec_std,
            "layout": "HWC -> CHW + batch_dim",
            "output_shape": list(tensor.shape),
            "note": (
                "input_mean e input_std fueron leídos del modelo en runtime "
                "(rec_model.input_mean / input_std), no hardcodeados. El cliente "
                "JS DEBE usar exactamente estos valores."
            ),
        },
        "sanity_check_manual_vs_get_feat": {
            "cosine_similarity": float(sim_manual_vs_ref),
            "max_abs_diff": max_abs_diff,
            "passed": bool(sim_manual_vs_ref >= 0.9999),
        },
        "success_criteria_for_js_client": {
            "cosine_similarity_min": 0.9999,
            "max_abs_diff_max": 1e-3,
        },
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[7/7] Fixture generado en {output_dir}/:")
    for p in (png_path, tensor_path, emb_path, meta_path):
        sz = p.stat().st_size
        print(f"        - {p.name:35s}  ({sz:>10,} bytes)")

    print()
    print("[OK] Fixture listo. Próximo paso: spike JS en cliente "
          "(sub-pasos D y E del Track 2a).")


if __name__ == "__main__":
    main()
