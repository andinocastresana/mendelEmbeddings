# =========================================
# ID: PHYLOFACE_EVALUATE_FULLFACE_MULTITASK
# VERSION: v0.1
# =========================================
# Tarea #29: alternativas full-face de baja capacidad tras el MLP fallido.
#
# Objetivo: probar si hay senal incremental sobre el coseno global antes de pasar
# a regiones. La idea "CCMTL-lite" toma del SoTA multi-task la intuicion de
# compartir informacion entre FS/FD/MS/MD, pero evita redes profundas: usa
# regresion logistica con features de score global y terminos por relacion.
#
# Ejecucion recomendada:
#   ./scripts/test-monitored.sh /home/diego/miniconda3/bin/conda run -n face-sim \
#     python scripts/evaluate_fullface_multitask.py --dataset KinFaceW-I

import argparse
import glob
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phyloface.benchmark import calibration, kinfacew  # noqa: E402
from phyloface.core.embedder import load_recognition_only, extract_embedding_from_aligned  # noqa: E402
from phyloface.core.metrics import cosine_similarity, euclidean_distance  # noqa: E402

REC_MODEL_PATH = Path.home() / ".insightface/models/buffalo_l/w600k_r50.onnx"
MODEL_VERSION = "w600k_r50"
REL_CODES = ["FS", "MD", "FD", "MS"]
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
        temp = read_max_temp_c()
        tstr = f"{temp:.0f}C" if temp is not None else "n/d"
        print(f"  embeds {relation}: {done}/{len(pending)} temp={tstr}", flush=True)
        if temp is not None and temp >= cool_threshold and done < len(pending):
            print(f"  [throttle] temp >= {cool_threshold}C -> pausa {cool_secs}s", flush=True)
            time.sleep(cool_secs)


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


def auc_or_nan(y_true, y_score) -> float:
    if len(set(map(int, y_true))) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def feature_vector(row: dict, spec: str) -> np.ndarray:
    cosine = float(row["cosine"])
    euclidean = float(row["euclidean"])
    rel = row["relation"]
    onehot = np.array([1.0 if rel == code else 0.0 for code in REL_CODES[1:]], dtype=np.float32)

    if spec == "cosine":
        return np.array([cosine], dtype=np.float32)
    if spec == "cosine_euclidean":
        return np.array([cosine, euclidean], dtype=np.float32)
    if spec == "shared_offsets":
        return np.concatenate(([cosine, euclidean], onehot)).astype(np.float32)
    if spec == "shared_slopes":
        interactions = []
        for code in REL_CODES[1:]:
            flag = 1.0 if rel == code else 0.0
            interactions.extend([cosine * flag, euclidean * flag])
        return np.concatenate(([cosine, euclidean], onehot, interactions)).astype(np.float32)
    raise ValueError(f"feature spec desconocido: {spec}")


def fit_predict_logreg(train_rows, test_rows, spec: str, c_value: float, seed: int):
    x_train = np.stack([feature_vector(r, spec) for r in train_rows])
    y_train = np.array([r["label"] for r in train_rows], dtype=int)
    x_test = np.stack([feature_vector(r, spec) for r in test_rows])
    y_test = np.array([r["label"] for r in test_rows], dtype=int)
    if len(np.unique(y_train)) < 2:
        return y_test, np.full(len(y_test), np.nan, dtype=float)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            solver="liblinear",
            max_iter=1000,
            random_state=seed,
        ),
    )
    clf.fit(x_train, y_train)
    return y_test, clf.predict_proba(x_test)[:, 1]


def evaluate_logreg(rows, spec: str, c_value: float, seed: int, relation: str | None = None) -> dict:
    scope = [r for r in rows if relation is None or r["relation"] == relation]
    folds = sorted({int(r["fold"]) for r in scope})
    fold_stats = []
    y_all, prob_all = [], []
    for fold in folds:
        train = [r for r in rows if int(r["fold"]) != fold and (relation is None or r["relation"] == relation)]
        test = [r for r in rows if int(r["fold"]) == fold and (relation is None or r["relation"] == relation)]
        if not train or not test:
            continue
        y_test, prob = fit_predict_logreg(train, test, spec, c_value, seed + fold)
        if np.isnan(prob).all():
            continue
        pred = (prob >= 0.5).astype(int)
        fold_stats.append({
            "fold": fold,
            "n_test": int(len(test)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "auc": auc_or_nan(y_test, prob),
        })
        y_all.extend(y_test.tolist())
        prob_all.extend(prob.tolist())

    labels = np.array([r["label"] for r in scope], dtype=int)
    accs = np.array([f["accuracy"] for f in fold_stats], dtype=float)
    aucs = np.array([f["auc"] for f in fold_stats], dtype=float)
    return {
        "n_pos": int((labels == 1).sum()),
        "n_neg": int((labels == 0).sum()),
        "n_folds": int(len(fold_stats)),
        "accuracy_mean": float(np.nanmean(accs)) if len(accs) else float("nan"),
        "accuracy_std": float(np.nanstd(accs)) if len(accs) else float("nan"),
        "auc_mean": float(np.nanmean(aucs)) if len(aucs) else float("nan"),
        "auc_std": float(np.nanstd(aucs)) if len(aucs) else float("nan"),
        "auc_oof": auc_or_nan(y_all, prob_all) if y_all else float("nan"),
        "folds": fold_stats,
    }


def evaluate_youden(rows, metric: str, higher_is_kin: bool, relation: str | None = None) -> dict:
    scope = [r for r in rows if relation is None or r["relation"] == relation]
    pairs = [
        {"fold": int(r["fold"]), "label": int(r["label"]), "score": float(r[metric])}
        for r in scope
    ]
    return calibration.cross_val_calibrate(pairs, higher_is_kin=higher_is_kin)


def build_rows(dataset: str, zip_path: Path, limit: int, batch_size: int,
               cool_threshold: float, cool_secs: float) -> list[dict]:
    print(f"Cargando solo modelo de reconocimiento ({REC_MODEL_PATH.name})...")
    rec_model = load_recognition_only(REC_MODEL_PATH)
    print("rec_model OK\n")

    rows = []
    emb_cache = {}
    with zipfile.ZipFile(zip_path) as zf:
        for rel in ["fs", "md", "fd", "ms"]:
            pairs = stratified_limit_pairs(kinfacew.load_pairs(zf, dataset, rel), limit)
            label = kinfacew.RELATIONS[rel][2]
            print(f"[{label}] {len(pairs)} pares; computando scores full-face...")
            names = kinfacew.unique_image_names(pairs)
            embed_images(zf, dataset, rel, names, rec_model, emb_cache,
                         batch_size, cool_threshold, cool_secs)
            for p in pairs:
                e1 = emb_cache.get((rel, p.name1))
                e2 = emb_cache.get((rel, p.name2))
                if e1 is None or e2 is None:
                    continue
                rows.append({
                    "relation": REL_CODE_UP[rel],
                    "fold": int(p.fold),
                    "label": int(p.label),
                    "cosine": float(cosine_similarity(e1, e2)),
                    "euclidean": float(euclidean_distance(e1, e2)),
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="KinFaceW-I")
    ap.add_argument("--zip", default=None, help="ruta al zip (default: data/input/datasets/<dataset>.zip)")
    ap.add_argument("--limit", type=int, default=0, help="cap de pares por relacion (0 = todos)")
    ap.add_argument("--c", type=float, default=0.25, help="regularizacion L2: C de LogisticRegression")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=120)
    ap.add_argument("--cool-threshold", type=float, default=85.0)
    ap.add_argument("--cool-secs", type=float, default=6.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    zip_path = Path(args.zip) if args.zip else PROJECT_ROOT / f"data/input/datasets/{args.dataset}.zip"
    if not zip_path.exists():
        print(f"FALTA el zip: {zip_path}")
        sys.exit(1)
    if not REC_MODEL_PATH.exists():
        print(f"FALTA el modelo de reconocimiento: {REC_MODEL_PATH}")
        sys.exit(1)

    print(
        f"Full-face multi-task #29 — dataset={args.dataset} "
        f"limit={args.limit or 'todos'} C={args.c}"
    )
    rows = build_rows(args.dataset, zip_path, args.limit, args.batch_size,
                      args.cool_threshold, args.cool_secs)

    results = {
        "baseline_youden_cosine": {},
        "baseline_youden_euclidean": {},
        "logreg_global_cosine": {},
        "logreg_global_cosine_euclidean": {},
        "logreg_shared_relation_offsets": {},
        "logreg_shared_relation_slopes": {},
        "logreg_per_relation_cosine_euclidean": {},
    }
    for rel in REL_CODES:
        results["baseline_youden_cosine"][rel] = evaluate_youden(rows, "cosine", True, rel)
        results["baseline_youden_euclidean"][rel] = evaluate_youden(rows, "euclidean", False, rel)
        results["logreg_per_relation_cosine_euclidean"][rel] = evaluate_logreg(
            rows, "cosine_euclidean", args.c, args.seed, rel
        )

    results["baseline_youden_cosine"]["ALL"] = evaluate_youden(rows, "cosine", True, None)
    results["baseline_youden_euclidean"]["ALL"] = evaluate_youden(rows, "euclidean", False, None)
    results["logreg_global_cosine"]["ALL"] = evaluate_logreg(rows, "cosine", args.c, args.seed, None)
    results["logreg_global_cosine_euclidean"]["ALL"] = evaluate_logreg(
        rows, "cosine_euclidean", args.c, args.seed, None
    )
    results["logreg_shared_relation_offsets"]["ALL"] = evaluate_logreg(
        rows, "shared_offsets", args.c, args.seed, None
    )
    results["logreg_shared_relation_slopes"]["ALL"] = evaluate_logreg(
        rows, "shared_slopes", args.c, args.seed, None
    )

    print(f"\n{'modelo':<38} {'rel':<4} {'acc':>8} {'auc/oof':>8}")
    print("-" * 64)
    order = [
        ("baseline_youden_cosine", "auc"),
        ("baseline_youden_euclidean", "auc"),
        ("logreg_global_cosine", "auc_oof"),
        ("logreg_global_cosine_euclidean", "auc_oof"),
        ("logreg_shared_relation_offsets", "auc_oof"),
        ("logreg_shared_relation_slopes", "auc_oof"),
    ]
    for model, auc_key in order:
        r = results[model]["ALL"]
        print(f"{model:<38} {'ALL':<4} {r['accuracy_mean']:>8.3f} {r[auc_key]:>8.3f}")
    for rel in REL_CODES:
        r = results["logreg_per_relation_cosine_euclidean"][rel]
        print(f"{'logreg_per_relation_cosine_euclidean':<38} {rel:<4} {r['accuracy_mean']:>8.3f} {r['auc_oof']:>8.3f}")

    artifact = {
        "v": 1,
        "computedAt": int(time.time() * 1000),
        "dataset": args.dataset,
        "modelVersion": MODEL_VERSION,
        "protocol": "5fold-cv-official",
        "task": 29,
        "experiment": "fullface-ccmtl-lite",
        "featureSpec": {
            "baseline_youden_cosine": "cosine threshold selected by Youden J on train folds",
            "baseline_youden_euclidean": "L2-normalized euclidean threshold selected by Youden J on train folds",
            "logreg_global_cosine": "single shared logistic model over cosine",
            "logreg_global_cosine_euclidean": "single shared logistic model over cosine+euclidean",
            "logreg_shared_relation_offsets": "shared logistic model over cosine+euclidean plus relation offsets",
            "logreg_shared_relation_slopes": "shared logistic model over cosine+euclidean plus relation offsets and score interactions",
            "logreg_per_relation_cosine_euclidean": "independent tiny logistic model per relation",
        },
        "regularizationC": args.c,
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
    out = Path(args.out) if args.out else PROJECT_ROOT / f"data/output/calibration/{args.dataset}_fullface_multitask.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2))
    print(f"\nArtefacto escrito: {out}")


if __name__ == "__main__":
    main()
