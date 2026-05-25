# =========================================
# ID: PHYLOFACE_RUN_CALIBRATION_KINFACEW
# VERSION: v1.0
# =========================================
# Fase A de la Tarea #6: calibración de umbrales de parentesco sobre KinFaceW.
#
# Pipeline:
#   1. Carga SOLO el modelo de reconocimiento (load_recognition_only) — sin los
#      submodelos de detección/landmark/genderage (menos RAM/calor).
#   2. Por relación (FS/MD/FD/MS): parsea los pares oficiales del .mat (folds +
#      kin/non-kin), computa embeddings de las imágenes únicas EN BATCHES con
#      enfriamiento adaptativo a temperatura, y cachea por nombre.
#   3. Calcula coseno y euclídea (L2-norm) por par.
#   4. Calibra con 5-fold CV (umbral de Youden ajustado en train, accuracy en
#      test) + AUC pooled, por relación y agregado (ALL).
#   5. Emite un artefacto JSON versionado (contrato hacia la web, Fase B) con
#      umbrales + accuracy + AUC + histogramas pre-binneados.
#
# Monitoreo: correr SIEMPRE vía el wrapper de recursos:
#   ./scripts/test-monitored.sh python3 scripts/run_calibration_kinfacew.py
#   (+ --limit N para un smoke rápido antes de la corrida completa)
#
# Batching adaptativo: tras cada batch, si la temp del sistema (sysfs
# thermal_zone) supera el umbral, pausa para enfriar. Defaults conservadores
# porque el spike tocó 94°C.

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

from phyloface.core.embedder import load_recognition_only, extract_embedding_from_aligned  # noqa: E402
from phyloface.core.metrics import cosine_similarity, euclidean_distance  # noqa: E402
from phyloface.benchmark import kinfacew, calibration  # noqa: E402

REC_MODEL_PATH = Path.home() / ".insightface/models/buffalo_l/w600k_r50.onnx"
MODEL_VERSION = "w600k_r50"
REL_CODE_UP = {"fs": "FS", "md": "MD", "fd": "FD", "ms": "MS"}
KINFACEW_II_WARNING = (
    "KinFaceW-II se reporta solo como referencia secundaria: sus pares "
    "positivos pueden provenir de la misma foto familiar, introduciendo senales "
    "compartidas de captura/contexto que no son parentesco facial. KinFaceW-I "
    "debe usarse como evaluacion primaria."
)


def dataset_warning(dataset: str) -> str | None:
    return KINFACEW_II_WARNING if dataset.lower() == "kinfacew-ii" else None


def read_max_temp_c():
    """Temp máxima entre las thermal zones de sysfs (°C), o None si no hay."""
    vals = []
    for f in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            with open(f) as fh:
                vals.append(int(fh.read().strip()) / 1000.0)
        except Exception:
            pass
    return max(vals) if vals else None


def embed_images(zf, dataset, relation, names, rec_model, cache,
                 batch_size, cool_threshold, cool_secs):
    """Computa embeddings de `names` (de una relación) en batches con
    enfriamiento adaptativo. Llena `cache[(relation, name)] = emb`."""
    pending = [n for n in names if (relation, n) not in cache]
    for i in range(0, len(pending), batch_size):
        batch = pending[i:i + batch_size]
        for name in batch:
            try:
                rgb = kinfacew.decode_aligned_rgb(zf, dataset, relation, name)
                cache[(relation, name)] = extract_embedding_from_aligned(rec_model, rgb)
            except Exception as e:
                print(f"  [skip] {relation}/{name}: {e}")
                cache[(relation, name)] = None
        t = read_max_temp_c()
        done = min(i + batch_size, len(pending))
        tstr = f"{t:.0f}°C" if t is not None else "n/d"
        print(f"  embeds {relation}: {done}/{len(pending)}  temp={tstr}", flush=True)
        if t is not None and t >= cool_threshold and done < len(pending):
            print(f"  [throttle] temp ≥ {cool_threshold}°C → pausa {cool_secs}s", flush=True)
            time.sleep(cool_secs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="KinFaceW-I")
    ap.add_argument("--zip", default=None, help="ruta al zip (default: data/input/datasets/<dataset>.zip)")
    ap.add_argument("--limit", type=int, default=0, help="cap de pares por relación (0 = todos); para smoke")
    ap.add_argument("--batch-size", type=int, default=120)
    ap.add_argument("--cool-threshold", type=float, default=85.0, help="°C: por encima, pausa entre batches")
    ap.add_argument("--cool-secs", type=float, default=6.0)
    ap.add_argument("--out", default=None, help="ruta del JSON de salida")
    args = ap.parse_args()

    zip_path = Path(args.zip) if args.zip else PROJECT_ROOT / f"data/input/datasets/{args.dataset}.zip"
    if not zip_path.exists():
        print(f"FALTA el zip: {zip_path}")
        sys.exit(1)
    if not REC_MODEL_PATH.exists():
        print(f"FALTA el modelo de reconocimiento: {REC_MODEL_PATH}")
        sys.exit(1)

    print(f"Calibración #6 — dataset={args.dataset} zip={zip_path.name} limit={args.limit or 'todos'}")
    print(f"Cargando solo modelo de reconocimiento ({REC_MODEL_PATH.name})...")
    rec_model = load_recognition_only(REC_MODEL_PATH)
    print("rec_model OK\n")

    metrics_def = [
        ("cosine", cosine_similarity, True),       # mayor = más kin
        ("euclidean", euclidean_distance, False),  # menor = más kin (L2-norm)
    ]
    # Acumuladores: artifact["cosine"]["FS"] = {...}; y "ALL" pooled.
    artifact_metrics = {m: {} for m, _, _ in metrics_def}
    pooled = {m: [] for m, _, _ in metrics_def}  # lista de dicts {fold,label,score}
    pooled_scores = {m: {"pos": [], "neg": []} for m, _, _ in metrics_def}

    emb_cache = {}
    summary_rows = []

    with zipfile.ZipFile(zip_path) as zf:
        for rel in ["fs", "md", "fd", "ms"]:
            pairs = kinfacew.load_pairs(zf, args.dataset, rel)
            if args.limit:
                pairs = pairs[: args.limit]
            label = kinfacew.RELATIONS[rel][2]
            print(f"[{label}] {len(pairs)} pares; computando embeddings...")
            names = kinfacew.unique_image_names(pairs)
            embed_images(zf, args.dataset, rel, names, rec_model, emb_cache,
                         args.batch_size, args.cool_threshold, args.cool_secs)

            # Scores por par y por métrica.
            per_metric_pairs = {m: [] for m, _, _ in metrics_def}
            per_metric_scores = {m: {"pos": [], "neg": []} for m, _, _ in metrics_def}
            for p in pairs:
                e1 = emb_cache.get((rel, p.name1))
                e2 = emb_cache.get((rel, p.name2))
                if e1 is None or e2 is None:
                    continue
                for m, fn, _ in metrics_def:
                    s = float(fn(e1, e2))
                    per_metric_pairs[m].append({"fold": p.fold, "label": p.label, "score": s})
                    per_metric_scores[m]["pos" if p.label == 1 else "neg"].append(s)

            for m, _, hik in metrics_def:
                stats = calibration.cross_val_calibrate(per_metric_pairs[m], higher_is_kin=hik)
                stats["histogram"] = calibration.histogram(
                    np.array(per_metric_scores[m]["pos"]),
                    np.array(per_metric_scores[m]["neg"]),
                )
                artifact_metrics[m][REL_CODE_UP[rel]] = stats
                pooled[m].extend(per_metric_pairs[m])
                pooled_scores[m]["pos"].extend(per_metric_scores[m]["pos"])
                pooled_scores[m]["neg"].extend(per_metric_scores[m]["neg"])
                if m == "cosine":
                    summary_rows.append((label, stats["n_pos"], stats["accuracy_mean"],
                                         stats["threshold_mean"], stats["auc"]))

    # Agregado ALL por métrica.
    for m, _, hik in metrics_def:
        stats = calibration.cross_val_calibrate(pooled[m], higher_is_kin=hik)
        stats["histogram"] = calibration.histogram(
            np.array(pooled_scores[m]["pos"]), np.array(pooled_scores[m]["neg"]))
        artifact_metrics[m]["ALL"] = stats

    # ---- Resumen en consola (coseno) ----
    print(f"\n{'relación':<22} {'n_pos':>6} {'acc(5cv)':>10} {'thr':>8} {'AUC':>6}")
    print("-" * 56)
    for label, npos, acc, thr, auc in summary_rows:
        print(f"{label:<22} {npos:>6} {acc:>10.3f} {thr:>8.3f} {auc:>6.3f}")
    allc = artifact_metrics["cosine"]["ALL"]
    print("-" * 56)
    print(f"{'ALL (coseno)':<22} {allc['n_pos']:>6} {allc['accuracy_mean']:>10.3f} "
          f"{allc['threshold_mean']:>8.3f} {allc['auc']:>6.3f}")

    # ---- Artefacto JSON ----
    artifact = {
        "v": 1,
        "computedAt": int(time.time() * 1000),
        "modelVersion": MODEL_VERSION,
        "dataset": args.dataset,
        "protocol": "5fold-cv-official",
        "primaryDataset": "KinFaceW-I",
        "evaluationRole": "primary" if args.dataset == "KinFaceW-I" else "secondary-biased",
        "warning": dataset_warning(args.dataset),
        "note": (
            "KinFaceW-I es la evaluacion primaria. KinFaceW-II tiene sesgo "
            "same-photo y debe reportarse solo como referencia secundaria. "
            "Ver _meta/BIBLIOGRAFIA_KINSHIP_DATASETS.md"
        ),
        "limit": args.limit or None,
        "metrics": artifact_metrics,
    }
    out = Path(args.out) if args.out else PROJECT_ROOT / f"data/output/calibration/{args.dataset}_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2))
    print(f"\nArtefacto escrito: {out}")


if __name__ == "__main__":
    main()
