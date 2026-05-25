# =========================================
# ID: PHYLOFACE_TRAIN_KINSHIP_MLP
# VERSION: v0.1
# =========================================
# Experimento de cabeza MLP para parentesco sobre embeddings ArcFace (Tarea #6,
# mejora posterior al baseline de umbral). Mantiene el protocolo honesto:
# folds oficiales de KinFaceW, sin fuga train/test.
#
# Feature por par:
#   [abs(e1 - e2), e1 * e2, cosine(e1,e2), euclidean(e1,e2)]
# donde e1/e2 son embeddings w600k_r50 L2-normalizados por las metricas del
# paquete. La cabeza es pequena y portable en principio a ONNX, pero este script
# solo evalua si hay senal incremental antes de exportar nada.
#
# Ejecucion recomendada (monitoreada):
#   ./scripts/test-monitored.sh /home/diego/miniconda3/bin/conda run -n face-sim \
#     python scripts/train_kinship_mlp.py --dataset KinFaceW-I
#
# Smoke rapido:
#   /home/diego/miniconda3/bin/conda run -n face-sim \
#     python scripts/train_kinship_mlp.py --limit 40 --max-iter 80

import argparse
import glob
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phyloface.benchmark import kinfacew  # noqa: E402
from phyloface.core.embedder import load_recognition_only, extract_embedding_from_aligned  # noqa: E402
from phyloface.core.metrics import cosine_similarity, euclidean_distance  # noqa: E402

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


def embed_images(zf, dataset, relation, names, rec_model, cache,
                 batch_size, cool_threshold, cool_secs):
    pending = [n for n in names if (relation, n) not in cache]
    for i in range(0, len(pending), batch_size):
        batch = pending[i:i + batch_size]
        for name in batch:
            try:
                rgb = kinfacew.decode_aligned_rgb(zf, dataset, relation, name)
                cache[(relation, name)] = extract_embedding_from_aligned(rec_model, rgb)
            except Exception as e:
                print(f"  [skip] {relation}/{name}: {e}", flush=True)
                cache[(relation, name)] = None
        done = min(i + batch_size, len(pending))
        t = read_max_temp_c()
        tstr = f"{t:.0f}C" if t is not None else "n/d"
        print(f"  embeds {relation}: {done}/{len(pending)} temp={tstr}", flush=True)
        if t is not None and t >= cool_threshold and done < len(pending):
            print(f"  [throttle] temp >= {cool_threshold}C -> pausa {cool_secs}s", flush=True)
            time.sleep(cool_secs)


def pair_features(e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    e1 = np.asarray(e1, dtype=np.float32)
    e2 = np.asarray(e2, dtype=np.float32)
    aux = np.array([
        float(cosine_similarity(e1, e2)),
        float(euclidean_distance(e1, e2)),
    ], dtype=np.float32)
    return np.concatenate([np.abs(e1 - e2), e1 * e2, aux]).astype(np.float32)


def stratified_limit_pairs(pairs, limit: int):
    """Submuestreo deterministico por fold+label para smokes no degenerados."""
    if not limit or limit >= len(pairs):
        return pairs
    groups = {}
    for p in pairs:
        groups.setdefault((int(p.fold), int(p.label)), []).append(p)
    keys = sorted(groups)
    selected = []
    idx = 0
    while len(selected) < limit:
        progressed = False
        for k in keys:
            bucket = groups[k]
            if idx < len(bucket):
                selected.append(bucket[idx])
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
        idx += 1
    return sorted(selected, key=lambda p: (int(p.fold), int(p.label), p.name1, p.name2))


def auc_or_nan(y_true, y_score) -> float:
    if len(set(map(int, y_true))) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def evaluate_group(rows, hidden_layers, max_iter, seed) -> dict:
    folds = sorted({int(r["fold"]) for r in rows})
    fold_stats = []
    y_all = np.array([r["label"] for r in rows], dtype=int)
    if len(np.unique(y_all)) < 2:
        return {
            "n_pos": int((y_all == 1).sum()),
            "n_neg": int((y_all == 0).sum()),
            "n_folds": 0,
            "accuracy_mean": float("nan"),
            "accuracy_std": float("nan"),
            "auc_mean": float("nan"),
            "auc_std": float("nan"),
            "folds": [],
        }

    for fold in folds:
        train = [r for r in rows if int(r["fold"]) != fold]
        test = [r for r in rows if int(r["fold"]) == fold]
        y_train = np.array([r["label"] for r in train], dtype=int)
        y_test = np.array([r["label"] for r in test], dtype=int)
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        x_train = np.stack([r["features"] for r in train])
        x_test = np.stack([r["features"] for r in test])
        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation="relu",
                solver="adam",
                alpha=1e-3,
                learning_rate_init=1e-3,
                max_iter=max_iter,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=seed + fold,
            ),
        )
        clf.fit(x_train, y_train)
        prob = clf.predict_proba(x_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        fold_stats.append({
            "fold": fold,
            "n_test": int(len(test)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "auc": auc_or_nan(y_test, prob),
        })

    accs = np.array([f["accuracy"] for f in fold_stats], dtype=float)
    aucs = np.array([f["auc"] for f in fold_stats], dtype=float)
    return {
        "n_pos": int((y_all == 1).sum()),
        "n_neg": int((y_all == 0).sum()),
        "n_folds": len(fold_stats),
        "accuracy_mean": float(np.nanmean(accs)) if len(accs) else float("nan"),
        "accuracy_std": float(np.nanstd(accs)) if len(accs) else float("nan"),
        "auc_mean": float(np.nanmean(aucs)) if len(aucs) else float("nan"),
        "auc_std": float(np.nanstd(aucs)) if len(aucs) else float("nan"),
        "folds": fold_stats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="KinFaceW-I")
    ap.add_argument("--zip", default=None, help="ruta al zip (default: data/input/datasets/<dataset>.zip)")
    ap.add_argument("--limit", type=int, default=0, help="cap de pares por relacion (0 = todos)")
    ap.add_argument("--hidden", default="64,32", help="capas ocultas, ej. 64,32")
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=120)
    ap.add_argument("--cool-threshold", type=float, default=85.0)
    ap.add_argument("--cool-secs", type=float, default=6.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    hidden_layers = tuple(int(x) for x in args.hidden.split(",") if x.strip())
    zip_path = Path(args.zip) if args.zip else PROJECT_ROOT / f"data/input/datasets/{args.dataset}.zip"
    if not zip_path.exists():
        print(f"FALTA el zip: {zip_path}")
        sys.exit(1)
    if not REC_MODEL_PATH.exists():
        print(f"FALTA el modelo de reconocimiento: {REC_MODEL_PATH}")
        sys.exit(1)

    print(
        f"MLP kinship #6 — dataset={args.dataset} limit={args.limit or 'todos'} "
        f"hidden={hidden_layers} max_iter={args.max_iter}"
    )
    print(f"Cargando solo modelo de reconocimiento ({REC_MODEL_PATH.name})...")
    rec_model = load_recognition_only(REC_MODEL_PATH)
    print("rec_model OK\n")

    emb_cache = {}
    rows_by_rel = {}
    all_rows = []
    with zipfile.ZipFile(zip_path) as zf:
        for rel in ["fs", "md", "fd", "ms"]:
            pairs = stratified_limit_pairs(kinfacew.load_pairs(zf, args.dataset, rel), args.limit)
            label = kinfacew.RELATIONS[rel][2]
            print(f"[{label}] {len(pairs)} pares; preparando embeddings/features...")
            names = kinfacew.unique_image_names(pairs)
            embed_images(zf, args.dataset, rel, names, rec_model, emb_cache,
                         args.batch_size, args.cool_threshold, args.cool_secs)

            rows = []
            for p in pairs:
                e1 = emb_cache.get((rel, p.name1))
                e2 = emb_cache.get((rel, p.name2))
                if e1 is None or e2 is None:
                    continue
                row = {
                    "relation": REL_CODE_UP[rel],
                    "fold": int(p.fold),
                    "label": int(p.label),
                    "features": pair_features(e1, e2),
                }
                rows.append(row)
                all_rows.append(row)
            rows_by_rel[REL_CODE_UP[rel]] = rows

    results = {}
    for rel, rows in rows_by_rel.items():
        print(f"Entrenando/evaluando MLP {rel} ({len(rows)} pares)...", flush=True)
        results[rel] = evaluate_group(rows, hidden_layers, args.max_iter, args.seed)
    print(f"Entrenando/evaluando MLP ALL ({len(all_rows)} pares)...", flush=True)
    results["ALL"] = evaluate_group(all_rows, hidden_layers, args.max_iter, args.seed)

    print(f"\n{'rel':<6} {'n_pos':>6} {'n_neg':>6} {'acc':>9} {'auc':>9}")
    print("-" * 42)
    for rel in ["FS", "MD", "FD", "MS", "ALL"]:
        r = results[rel]
        print(
            f"{rel:<6} {r['n_pos']:>6} {r['n_neg']:>6} "
            f"{r['accuracy_mean']:>9.3f} {r['auc_mean']:>9.3f}"
        )

    artifact = {
        "v": 1,
        "computedAt": int(time.time() * 1000),
        "dataset": args.dataset,
        "modelVersion": MODEL_VERSION,
        "protocol": "5fold-cv-official",
        "head": "sklearn-mlp",
        "featureSpec": "absdiff512+prod512+cosine+euclidean",
        "hiddenLayers": list(hidden_layers),
        "maxIter": args.max_iter,
        "seed": args.seed,
        "limit": args.limit or None,
        "primaryDataset": "KinFaceW-I",
        "evaluationRole": "primary" if args.dataset == "KinFaceW-I" else "secondary-biased",
        "warning": (
            "KinFaceW-II se reporta solo como referencia secundaria por sesgo same-photo."
            if args.dataset.lower() == "kinfacew-ii" else None
        ),
        "results": results,
    }
    out = Path(args.out) if args.out else PROJECT_ROOT / f"data/output/calibration/{args.dataset}_mlp_head.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2))
    print(f"\nArtefacto escrito: {out}")


if __name__ == "__main__":
    main()
