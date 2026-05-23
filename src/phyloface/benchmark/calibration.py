# =========================================
# ID: PHYLOFACE_BENCHMARK_CALIBRATION
# VERSION: v1.0
# =========================================
# Lógica PURA de calibración de umbrales de parentesco (Tarea #6). Sin I/O, sin
# embeddings, sin deps pesadas (solo numpy) — recibe scores + labels y devuelve
# umbrales/accuracy/AUC/histogramas. Testeable en aislamiento.
#
# Convención de dirección de métrica:
#   - higher_is_kin=True  → score más ALTO = más parentesco (coseno).
#   - higher_is_kin=False → score más BAJO = más parentesco (euclídea sobre
#     vectores L2-normalizados).
#
# Por qué no sklearn: el surface que necesitamos (sweep de umbral por Youden,
# accuracy, AUC rank-based, binning de histograma) son ~40 LOC en numpy; evita
# una dep que puede no estar instalada y mantiene el módulo autocontenido. Si
# en el futuro se necesita ROC/PR completas o calibración de probabilidad,
# reevaluar incorporar sklearn.

import numpy as np


# -----------------------------------------
# AUC rank-based = P(score_kin "mejor" que score_nonkin). Threshold-free,
# resumen de separabilidad. Empates = 0.5. Para higher_is_kin=False invertimos
# el signo para que "mejor" siga significando "más kin".
# -----------------------------------------
def auc_rank(pos: np.ndarray, neg: np.ndarray, higher_is_kin: bool = True) -> float:
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    if not higher_is_kin:
        pos, neg = -pos, -neg
    wins = 0.0
    for p in pos:
        wins += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return wins / (len(pos) * len(neg))


def _predict_kin(scores: np.ndarray, thr: float, higher_is_kin: bool) -> np.ndarray:
    return (scores >= thr) if higher_is_kin else (scores <= thr)


# -----------------------------------------
# Umbral que maximiza el índice J de Youden (TPR - FPR) sobre (scores, labels).
# labels: 1 = kin, 0 = non-kin. Candidatos = midpoints entre scores únicos
# ordenados + extremos. Devuelve (thr, J).
# -----------------------------------------
def best_threshold_youden(scores: np.ndarray, labels: np.ndarray, higher_is_kin: bool = True):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = labels == 1
    neg = labels == 0
    n_pos = max(1, pos.sum())
    n_neg = max(1, neg.sum())

    uniq = np.unique(scores)
    if len(uniq) == 1:
        cands = np.array([uniq[0] - 1e-6, uniq[0] + 1e-6])
    else:
        mids = (uniq[:-1] + uniq[1:]) / 2.0
        cands = np.concatenate(([uniq[0] - 1e-6], mids, [uniq[-1] + 1e-6]))

    best_thr, best_j = cands[0], -np.inf
    for thr in cands:
        pred = _predict_kin(scores, thr, higher_is_kin)
        tpr = np.sum(pred & pos) / n_pos
        fpr = np.sum(pred & neg) / n_neg
        j = tpr - fpr
        if j > best_j:
            best_j, best_thr = j, thr
    return float(best_thr), float(best_j)


def accuracy_at(scores: np.ndarray, labels: np.ndarray, thr: float, higher_is_kin: bool = True) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pred = _predict_kin(scores, thr, higher_is_kin).astype(int)
    return float(np.mean(pred == labels))


# -----------------------------------------
# Calibración 5-fold (protocolo KinFaceW): para cada fold de TEST, ajustar el
# umbral de Youden sobre los OTROS folds (train) y medir accuracy en el de test.
# Devuelve dict con accuracy media±std, umbral medio, accuracies por fold y AUC
# pooled (sobre todos los pares, threshold-free).
#
# pairs: lista de dicts {fold:int, label:int(0/1), score:float}.
# -----------------------------------------
def cross_val_calibrate(pairs: list[dict], higher_is_kin: bool = True) -> dict:
    folds = sorted({p["fold"] for p in pairs})
    scores_all = np.array([p["score"] for p in pairs], dtype=float)
    labels_all = np.array([p["label"] for p in pairs], dtype=int)

    fold_accs, fold_thrs = [], []
    for f in folds:
        is_test = np.array([p["fold"] == f for p in pairs])
        is_train = ~is_test
        if is_train.sum() == 0 or is_test.sum() == 0:
            continue
        thr, _ = best_threshold_youden(scores_all[is_train], labels_all[is_train], higher_is_kin)
        acc = accuracy_at(scores_all[is_test], labels_all[is_test], thr, higher_is_kin)
        fold_accs.append(acc)
        fold_thrs.append(thr)

    pos = scores_all[labels_all == 1]
    neg = scores_all[labels_all == 0]
    return {
        "n_pos": int((labels_all == 1).sum()),
        "n_neg": int((labels_all == 0).sum()),
        "n_folds": len(fold_accs),
        "accuracy_mean": float(np.mean(fold_accs)) if fold_accs else float("nan"),
        "accuracy_std": float(np.std(fold_accs)) if fold_accs else float("nan"),
        "threshold_mean": float(np.mean(fold_thrs)) if fold_thrs else float("nan"),
        "fold_accuracies": [round(a, 4) for a in fold_accs],
        "auc": round(auc_rank(pos, neg, higher_is_kin), 4),
    }


# -----------------------------------------
# Histograma pre-binneado de las distribuciones pos/neg (para que la web dibuje
# barras + ubique el cosine propio sin necesitar los embeddings). Devuelve
# edges de bins + conteos. El rango por default cubre coseno [-? , ?]; lo
# tomamos del min/max observado con un pad.
# -----------------------------------------
def histogram(pos: np.ndarray, neg: np.ndarray, n_bins: int = 40) -> dict:
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    allv = np.concatenate([pos, neg]) if (len(pos) and len(neg)) else np.concatenate([pos, neg, [0.0]])
    lo, hi = float(np.min(allv)), float(np.max(allv))
    pad = (hi - lo) * 0.05 if hi > lo else 0.1
    edges = np.linspace(lo - pad, hi + pad, n_bins + 1)
    pos_counts, _ = np.histogram(pos, bins=edges)
    neg_counts, _ = np.histogram(neg, bins=edges)
    return {
        "bin_edges": [round(float(e), 5) for e in edges],
        "pos_counts": [int(c) for c in pos_counts],
        "neg_counts": [int(c) for c in neg_counts],
    }
