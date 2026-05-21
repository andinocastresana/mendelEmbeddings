#!/usr/bin/env python3
# =========================================
# ID: PHYLOFACE_SPIKE_004
# VERSION: v2.0
# =========================================
# Script de SPIKE para Track 2a — paridad del pipeline end-to-end JS vs Python.
#
# Cambio v1 -> v2 (2026-05-21, sesión multi-imagen):
# - Pasa de procesar UNA imagen única a procesar TODAS las imágenes de un
#   directorio (`data/input/img/spike_e2e_set/` por default).
# - Dedup por SHA-256: si una imagen ya fue procesada en una corrida anterior
#   y sigue presente en el set, no se recompute (ahorra tiempo en sets grandes).
# - Fixture multi-caso: `cases.json` con array de objetos {hash, ...} en lugar
#   de los archivos sueltos `reference_embedding.json` / `reference_detection.json`.
# - Imágenes públicas: copiadas a `client/public/spike_fixtures_detection/images/<hash>.png`
#   para que Vite las sirva al cliente. El hash como filename garantiza que no
#   haya colisiones por nombres distintos al mismo contenido.
# - Append-only a `_meta/spike_004_runs.md`: cada corrida deja una entrada con
#   timestamp, cuántas imágenes nuevas/reusadas, métricas de cada caso (Python
#   solo: bbox, det_score). Las métricas del JS NO van acá; las captura
#   el componente y/o el botón "descargar JSON" del SpikeDetection.tsx.
#
# Decisión: cleanup de imágenes huérfanas.
# Si quitás una imagen del set, su PNG queda en disco (regla del proyecto: no
# borrar). Pero deja de aparecer en `cases.json`, así que el cliente JS no la
# va a iterar más. Si querés "limpiar" PNGs huérfanos, hay que moverlos a
# `_toReview/` a mano (regla del proyecto).
#
# Decisión: el set inicial.
# Arranca con `BrunoFondoBlanco.jpeg` (la imagen del spike v1). Vos agregás
# más imágenes copiándolas al filesystem en `data/input/img/spike_e2e_set/`.
# A futuro (Track 2b — ver memoria `project-track2b-dataset-pipeline`), va a
# haber drag-and-drop browser + DB para que los usuarios contribuyan al set.
#
# Criterio de éxito del JS (sin cambios desde v1):
#   cosine_similarity(emb_py, emb_js) > 0.97 — por caso.
# El PASS GLOBAL del spike se da si TODOS los casos pasan; basta uno que falle
# para encender la alerta (al cliente lo decide; este script solo genera datos).

# -----------------------------------------
# FILE: scripts/verify_detection_web_parity.py
# -----------------------------------------

import argparse
import hashlib
import json
import shutil
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
from phyloface.core.embedder import (  # noqa: E402
    get_recognition_model,
    extract_embedding_from_aligned,
)


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}
RUNS_MD_RELPATH = "_meta/spike_004_runs.md"


# =========================================================
# Helpers
# =========================================================
def sha256_of_file(path: Path) -> str:
    """SHA-256 hex digest del contenido binario del archivo."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def list_images(image_dir: Path) -> list[Path]:
    """Lista ordenada (estable) de imágenes soportadas en el directorio."""
    return sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )


def load_previous_cases(cases_json_path: Path) -> dict[str, dict]:
    """
    Devuelve un dict {hash: case} con los cases de la corrida anterior, para
    dedup. Si no hay corrida anterior, devuelve dict vacío.
    """
    if not cases_json_path.exists():
        return {}
    try:
        data = json.loads(cases_json_path.read_text())
        return {c["hash"]: c for c in data.get("cases", [])}
    except (json.JSONDecodeError, KeyError, TypeError):
        # Si el archivo está corrupto/desactualizado, recomputar todo es seguro.
        print(f"[!] cases.json existente no se pudo parsear; ignorando para dedup.")
        return {}


def process_one_image(
    app, rec_model, image_path: Path, output_images_dir: Path,
) -> dict | None:
    """
    Corre el pipeline Python completo sobre una imagen y devuelve un dict
    con todo lo que el JS necesita para esa imagen.
    Si no detecta cara, devuelve None.
    """
    img_rgb = load_image(image_path)
    H, W = img_rgb.shape[:2]
    img_hash = sha256_of_file(image_path)

    _, face_records = detect_faces_in_image(
        app=app,
        img_rgb=img_rgb,
        photo_label=image_path.stem,
        pad_x=0.20,
        pad_y=0.35,
    )
    if len(face_records) == 0:
        return None

    face = face_records[0]
    aligned_rgb = align_face_from_record(
        face_record=face, image_size=112, margin_ratio=0.0,
    )
    emb_ref = extract_embedding_from_aligned(rec_model, aligned_rgb)

    # Copia/sobrescribe la imagen al directorio público del cliente. Usamos el
    # hash como filename (extensión .png siempre, re-codificada via cv2) — así
    # el JS pide /images/<hash>.png sin importar el nombre original.
    public_filename = f"{img_hash}.png"
    public_path = output_images_dir / public_filename
    output_images_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(public_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    return {
        "hash": img_hash,
        "source_filename": image_path.name,
        "public_filename": public_filename,
        "image_size_wh": [int(W), int(H)],
        "bbox_global": [int(v) for v in face["bbox"]],
        "det_score": float(face["det_score"]),
        "kps_global": {
            "values": face["kps"].astype(np.float32).tolist(),
            "order": "left_eye, right_eye, nose, left_mouth, right_mouth",
            "convention": "image-space (left=image left)",
        },
        "reference_embedding": {
            "shape": list(emb_ref.shape),
            "dtype": "float32",
            "norm_l2": float(np.linalg.norm(emb_ref)),
            "values": emb_ref.tolist(),
        },
    }


def append_run_to_md(
    md_path: Path,
    image_dir_rel: str,
    cases_in_set: list[dict],
    n_new: int,
    n_reused: int,
    set_hash: str,
) -> None:
    """Appendea una sección al MD acumulativo (lo crea si no existe)."""
    if not md_path.exists():
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(
            "# Spike #004 — corridas\n\n"
            "Append-only. Cada sección registra UNA corrida del script Python\n"
            "`scripts/verify_detection_web_parity.py`, que es el que mantiene\n"
            "el fixture multi-imagen del spike e2e.\n\n"
            "Solo se registran datos del lado Python (qué imágenes hay en el\n"
            "set, bbox/det_score detectados). Las métricas del cliente JS\n"
            "(cosine vs reference) se capturan por separado en el componente\n"
            "`SpikeDetection.tsx` y se exportan via su botón 'descargar JSON'\n"
            "si querés persistirlas a mano.\n\n"
            "---\n"
        )

    now_utc = datetime.utcnow().isoformat() + "Z"
    n_total = len(cases_in_set)

    lines = [
        f"\n## {now_utc}\n",
        f"- **Set dir**: `{image_dir_rel}`\n",
        f"- **Imágenes en set**: {n_total}  (nuevas: {n_new}, reusadas via dedup: {n_reused})\n",
        f"- **Set hash agregado**: `{set_hash[:16]}...`\n",
        f"- **Casos**:\n",
        "\n",
        "| # | hash (16ch) | source filename | det_score | bbox (x1,y1,x2,y2) |\n",
        "|---|-------------|------------------|-----------|---------------------|\n",
    ]
    for i, c in enumerate(cases_in_set, start=1):
        lines.append(
            f"| {i} | `{c['hash'][:16]}` | {c['source_filename']} | "
            f"{c['det_score']:.3f} | {tuple(c['bbox_global'])} |\n"
        )
    lines.append("\n---\n")

    with open(md_path, "a") as f:
        f.writelines(lines)


def compute_set_hash(cases: list[dict]) -> str:
    """Hash agregado de los hashes individuales (orden estable)."""
    h = hashlib.sha256()
    for c in sorted(cases, key=lambda c: c["hash"]):
        h.update(c["hash"].encode())
    return h.hexdigest()


# =========================================================
# Pipeline principal
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Genera fixture multi-imagen para el spike #004 e2e del Track 2a. "
            "Procesa todas las imágenes del directorio, dedup por SHA-256, "
            "appendea entrada a _meta/spike_004_runs.md."
        )
    )
    parser.add_argument(
        "--image-dir",
        default="data/input/img/spike_e2e_set",
        help="Directorio con las imágenes a procesar (relativo al root).",
    )
    parser.add_argument(
        "--output-dir",
        default="client/public/spike_fixtures_detection",
        help="Directorio donde guardar el fixture (relativo al root).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignorar dedup y reprocesar todas las imágenes (útil al cambiar el pipeline Python).",
    )
    args = parser.parse_args()

    image_dir = PROJECT_ROOT / args.image_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_images_dir = output_dir / "images"
    cases_json_path = output_dir / "cases.json"
    meta_path = output_dir / "metadata.json"
    md_path = PROJECT_ROOT / RUNS_MD_RELPATH

    if not image_dir.exists():
        print(f"[!] image-dir no existe: {image_dir}")
        sys.exit(1)

    images = list_images(image_dir)
    if not images:
        print(
            f"[!] No hay imágenes ({sorted(SUPPORTED_EXTS)}) en {image_dir}\n"
            f"    Agregá fotos a esa carpeta y volvé a correr el script."
        )
        sys.exit(1)

    print(f"[1/5] image-dir: {image_dir}")
    print(f"      output:    {output_dir}")
    print(f"      imágenes encontradas: {len(images)}")

    # Dedup: cargar cases.json previo.
    previous = {} if args.force else load_previous_cases(cases_json_path)
    if previous and not args.force:
        print(f"      cases.json previo: {len(previous)} entradas (dedup activo)")
    elif args.force:
        print(f"      --force: ignorando dedup, reprocesando todo")

    # InsightFace + recognition model.
    print("[2/5] Inicializando InsightFace...")
    app = init_face_app(providers=["CPUExecutionProvider"])
    rec_model = get_recognition_model(app)

    # Procesar imágenes.
    print(f"[3/5] Procesando {len(images)} imágenes...")
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_in_set: list[dict] = []
    n_new = 0
    n_reused = 0
    n_no_face = 0

    for i, img_path in enumerate(images, start=1):
        img_hash = sha256_of_file(img_path)
        if img_hash in previous and not args.force:
            # Reusar el case anterior tal cual, pero también garantizar que la
            # imagen pública esté presente (si alguien borró el dir output).
            case = previous[img_hash]
            public_path = output_images_dir / case["public_filename"]
            if not public_path.exists():
                output_images_dir.mkdir(parents=True, exist_ok=True)
                img_rgb = load_image(img_path)
                cv2.imwrite(str(public_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            cases_in_set.append(case)
            n_reused += 1
            print(f"  [{i}/{len(images)}] {img_path.name}  hash={img_hash[:12]}  REUSED")
        else:
            case = process_one_image(app, rec_model, img_path, output_images_dir)
            if case is None:
                n_no_face += 1
                print(f"  [{i}/{len(images)}] {img_path.name}  hash={img_hash[:12]}  NO FACE — skipped")
                continue
            cases_in_set.append(case)
            n_new += 1
            print(
                f"  [{i}/{len(images)}] {img_path.name}  hash={img_hash[:12]}  "
                f"NEW  det_score={case['det_score']:.3f}"
            )

    if not cases_in_set:
        print("[!] Ningún caso válido (todas las imágenes fallaron detección).")
        sys.exit(2)

    set_hash = compute_set_hash(cases_in_set)

    # === 4. Escribir cases.json + metadata.json ===
    print(f"[4/5] Escribiendo fixture ({len(cases_in_set)} casos)...")
    cases_payload = {
        "version": 2,
        "set_hash": set_hash,
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "cases": cases_in_set,
    }
    with open(cases_json_path, "w") as f:
        json.dump(cases_payload, f)

    metadata = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "version": 2,
        "image_dir": args.image_dir,
        "n_cases_total": len(cases_in_set),
        "n_cases_new": n_new,
        "n_cases_reused": n_reused,
        "n_cases_no_face": n_no_face,
        "set_hash": set_hash,
        "python_pipeline": {
            "detector": "InsightFace buffalo_l det_10g (SCRFD)",
            "alignment": {
                "image_size": 112,
                "margin_ratio": 0.0,
                "method": "face_align.estimate_norm + warpAffine BORDER_REPLICATE",
            },
            "embedding": {
                "model": "w600k_r50.onnx",
                "preprocessing": "(px - 127.5) / 127.5, HWC→CHW",
                "output_dim": 512,
            },
        },
        "js_pipeline_expected": {
            "detector": (
                "MediaPipe Face Mesh (Tasks for Web) con refine_landmarks=true. "
                "5 kps derivados de índices del mesh: 468, 473, 4, 61, 291 "
                "(orden InsightFace image-space). Índice de nariz 4 elegido "
                "empíricamente sobre 1 (ver SpikeDetection.tsx)."
            ),
            "alignment": "client/src/lib/alignment.ts (validado en spike #003)",
            "embedding": "onnxruntime-web + w600k_r50.onnx (validado en spike #001)",
        },
        "success_criteria_for_js_client": {
            "cosine_similarity_min": 0.97,
            "max_abs_diff_informative_only": True,
            "kps_mean_distance_informative_only": True,
            "global_pass_rule": "all cases must pass cosine_similarity_min individually",
            "note": (
                "PASS/FAIL único sobre cosine_similarity por caso. Se acumula "
                "GLOBAL PASS = todos los casos pasaron. Ver historia de "
                "threshold en _meta/spike_004_runs.md."
            ),
        },
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # === 5. Appendear al MD ===
    print(f"[5/5] Appendeando run a {md_path}...")
    append_run_to_md(
        md_path=md_path,
        image_dir_rel=args.image_dir,
        cases_in_set=cases_in_set,
        n_new=n_new,
        n_reused=n_reused,
        set_hash=set_hash,
    )

    # Resumen.
    print()
    print(f"[OK] Fixture multi-imagen listo en {output_dir}/")
    print(f"     casos totales: {len(cases_in_set)}  (nuevos: {n_new}, reusados: {n_reused}, no_face: {n_no_face})")
    print(f"     set_hash: {set_hash[:16]}...")
    for p in (cases_json_path, meta_path, md_path):
        rel = p.relative_to(PROJECT_ROOT) if p.is_absolute() else p
        sz = p.stat().st_size if p.exists() else 0
        print(f"     - {str(rel):50s}  ({sz:>10,} bytes)")
    print()
    print(
        "[NEXT] Refrescar la tab 'Spike detección (e2e)' en el browser para que "
        "el cliente cargue el nuevo cases.json y itere sobre todos los casos."
    )


if __name__ == "__main__":
    main()
