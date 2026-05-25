# =========================================
# ID: PHYLOFACE_VALIDATE_REGION_EMBEDDINGS_KINFACEW
# VERSION: v0.1
# =========================================
# Tarea #5: sanity check de embeddings por region contra KinFaceW.
#
# Este script es deliberadamente conservador: KinFaceW trae caras 64x64
# pre-recortadas; MediaPipe puede fallar en algunas. El objetivo no es cerrar el
# SoTA regional, sino medir si re-aplicar ArcFace a crops/máscaras produce una
# senal util o una metrica inestable antes de integrarlo al producto.

import argparse
import glob
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phyloface.benchmark import calibration, kinfacew  # noqa: E402
from phyloface.core.embedder import load_recognition_only  # noqa: E402
from phyloface.landmarks import init_face_mesh, get_face_mesh_landmarks  # noqa: E402
from phyloface.regions import (  # noqa: E402
    CANONICAL_REGION_NAMES,
    REGIONAL_EMBEDDINGS_VERSION,
    extract_regions_v2_masked,
    extract_region_embeddings,
    compare_region_embeddings,
)

REC_MODEL_PATH = Path.home() / ".insightface/models/buffalo_l/w600k_r50.onnx"
MODEL_VERSION = "w600k_r50"
REL_CODE_UP = {"fs": "FS", "md": "MD", "fd": "FD", "ms": "MS"}


def read_max_temp_c():
    vals = []
    for f in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            with open(f) as fh:
                vals.append(int(fh.read().strip()) / 1000.0)
        except Exception:
            pass
    return max(vals) if vals else None


def stratified_limit_pairs(pairs, limit: int):
    if not limit or limit >= len(pairs):
        return pairs
    groups = {}
    for p in pairs:
        groups.setdefault((int(p.fold), int(p.label)), []).append(p)
    selected = []
    idx = 0
    keys = sorted(groups)
    while len(selected) < limit:
        progressed = False
        for key in keys:
            bucket = groups[key]
            if idx < len(bucket):
                selected.append(bucket[idx])
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
        idx += 1
    return sorted(selected, key=lambda p: (int(p.fold), int(p.label), p.name1, p.name2))


def prepare_image_region_embeddings(zf, dataset, rel, name, face_mesh, rec_model, cache):
    key = (rel, name)
    if key in cache:
        return cache[key]
    try:
        rgb = kinfacew.decode_aligned_rgb(zf, dataset, rel, name)
        landmarks = get_face_mesh_landmarks(face_mesh, rgb)
        regions = extract_regions_v2_masked(rgb, landmarks)
        cache[key] = {
            "ok": True,
            "regions": extract_region_embeddings(rec_model, regions, crop_key="crop_masked_rgb"),
            "error": None,
        }
    except Exception as exc:
        cache[key] = {"ok": False, "regions": None, "error": str(exc)}
    return cache[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="KinFaceW-I")
    ap.add_argument("--zip", default=None)
    ap.add_argument("--limit", type=int, default=40, help="pares por relacion; 0 = todos")
    ap.add_argument("--cool-threshold", type=float, default=88.0)
    ap.add_argument("--cool-secs", type=float, default=6.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    zip_path = Path(args.zip) if args.zip else PROJECT_ROOT / f"data/input/datasets/{args.dataset}.zip"
    if not zip_path.exists():
        print(f"FALTA el zip: {zip_path}")
        sys.exit(1)
    if not REC_MODEL_PATH.exists():
        print(f"FALTA el modelo: {REC_MODEL_PATH}")
        sys.exit(1)

    print(
        f"Regional embeddings sanity #5 — dataset={args.dataset} "
        f"limit={args.limit or 'todos'} version={REGIONAL_EMBEDDINGS_VERSION}"
    )
    rec_model = load_recognition_only(REC_MODEL_PATH)
    face_mesh = init_face_mesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    cache = {}
    rows_by_region = {name: [] for name in CANONICAL_REGION_NAMES}
    failures = {}

    with zipfile.ZipFile(zip_path) as zf:
        for rel in ["fs", "md", "fd", "ms"]:
            pairs = stratified_limit_pairs(kinfacew.load_pairs(zf, args.dataset, rel), args.limit)
            print(f"[{REL_CODE_UP[rel]}] {len(pairs)} pares")
            for i, p in enumerate(pairs, 1):
                a = prepare_image_region_embeddings(zf, args.dataset, rel, p.name1, face_mesh, rec_model, cache)
                b = prepare_image_region_embeddings(zf, args.dataset, rel, p.name2, face_mesh, rec_model, cache)
                if not a["ok"]:
                    failures[(rel, p.name1)] = a["error"]
                if not b["ok"]:
                    failures[(rel, p.name2)] = b["error"]
                if a["ok"] and b["ok"]:
                    scores = compare_region_embeddings(a["regions"], b["regions"])
                    for region_name, item in scores.items():
                        if item["valid"] and not np.isnan(item["cosine"]):
                            rows_by_region[region_name].append({
                                "fold": int(p.fold),
                                "label": int(p.label),
                                "score": float(item["cosine"]),
                            })
                if i % 10 == 0:
                    temp = read_max_temp_c()
                    tstr = f"{temp:.0f}C" if temp is not None else "n/d"
                    print(f"  {REL_CODE_UP[rel]} {i}/{len(pairs)} temp={tstr}", flush=True)
                    if temp is not None and temp >= args.cool_threshold:
                        print(f"  [throttle] temp >= {args.cool_threshold}C -> pausa {args.cool_secs}s", flush=True)
                        time.sleep(args.cool_secs)

    results = {}
    print(f"\n{'region':<18} {'n_pos':>6} {'n_neg':>6} {'acc':>8} {'auc':>8}")
    print("-" * 54)
    for region_name in CANONICAL_REGION_NAMES:
        rows = rows_by_region[region_name]
        if rows:
            stats = calibration.cross_val_calibrate(rows, higher_is_kin=True)
        else:
            stats = {
                "n_pos": 0,
                "n_neg": 0,
                "n_folds": 0,
                "accuracy_mean": float("nan"),
                "accuracy_std": float("nan"),
                "threshold_mean": float("nan"),
                "fold_accuracies": [],
                "auc": float("nan"),
            }
        results[region_name] = stats
        print(
            f"{region_name:<18} {stats['n_pos']:>6} {stats['n_neg']:>6} "
            f"{stats['accuracy_mean']:>8.3f} {stats['auc']:>8.3f}"
        )

    artifact = {
        "v": 1,
        "computedAt": int(time.time() * 1000),
        "dataset": args.dataset,
        "limit": args.limit or None,
        "modelVersion": MODEL_VERSION,
        "regionalEmbeddingsVersion": REGIONAL_EMBEDDINGS_VERSION,
        "protocol": "5fold-cv-official",
        "warning": "Sanity check: ArcFace se re-aplica a parches regionales, no a rostros completos.",
        "nImageFailures": len(failures),
        "sampleFailures": [
            {"image": f"{rel}/{name}", "error": error}
            for (rel, name), error in list(failures.items())[:20]
        ],
        "results": results,
    }
    out = Path(args.out) if args.out else PROJECT_ROOT / f"data/output/calibration/{args.dataset}_region_embeddings_sanity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2))
    print(f"\nFallos de imagen: {len(failures)}")
    print(f"Artefacto escrito: {out}")


if __name__ == "__main__":
    main()
