#!/usr/bin/env python3
# =========================================
# ID: PHYLOFACE_TEAM_SIM_VS_GEO
# VERSION: v0.2
# =========================================
# FILE: scripts/plot_team_similarity_vs_geo.py
#
# Scatter parecido-facial vs distancia entre capitales, coloreado por lazo
# colonial / idioma compartido, para testear si el parecido facial entre
# selecciones se explica por geografía y/o colonialidad.
#
# Cruza la matriz facial equipo-equipo del payload de vitrina (FIFA por default,
# nombres en español -> resueltos por geo_team_resolve) con la distancia entre
# capitales + colonizador + idiomas de build_capitals_distance_matrix.py.
#
# "Lazo colonial" = mismo último colonizador (colonia-colonia) OR la selección B
# ES el colonizador de A (colonia-colonizador).
#
# Stats: Pearson/Spearman, Mantel (permutación) y correlaciones parciales que
# separan distancia de colonialidad.
#
# Uso:
#   conda run -n face-sim python scripts/plot_team_similarity_vs_geo.py
#   ... --facial <payload.json> --aggregation mean|median|top3_mean|top5_mean

import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geo_team_resolve import (make_resolver, haversine, colonial_link,
                              share_language, partial_corr)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_FACIAL = os.path.join(REPO_ROOT, "data", "output", "teams",
                              "vitrina_fifa_northamerica2026_similarity_pilot.json")
OUT_DIR = os.path.join(REPO_ROOT, "data", "output", "geo")


def mantel(sim, dist, perms=10000, seed=0):
    n = sim.shape[0]
    iu = np.triu_indices(n, 1)
    s, d = sim[iu], dist[iu]
    r_obs = np.corrcoef(s, d)[0, 1]
    rng = np.random.default_rng(seed)
    count = sum(1 for _ in range(perms)
                if abs(np.corrcoef(sim[np.ix_(p := rng.permutation(n), p)][iu], d)[0, 1]) >= abs(r_obs))
    return r_obs, (count + 1) / (perms + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facial", default=DEFAULT_FACIAL)
    ap.add_argument("--aggregation", default="mean",
                    choices=["mean", "median", "top3_mean", "top5_mean"])
    ap.add_argument("--perms", type=int, default=10000)
    ap.add_argument("--tag", default="fifa", help="sufijo de los archivos de salida.")
    args = ap.parse_args()

    fac = json.load(open(args.facial, encoding="utf-8"))
    teams = fac["teams"]
    mats = fac.get("team_similarity_matrices") or {}
    M = np.array(mats.get(args.aggregation) or fac["team_similarity_matrix"], dtype=float)
    resolve, _ = make_resolver()

    resolved = [resolve(t) for t in teams]
    missing = [t for t, g in zip(teams, resolved) if g is None]
    idx = [i for i, g in enumerate(resolved) if g is not None]
    if missing:
        print(f"[viz] sin geo (excluidos): {missing}")
    print(f"[viz] {len(idx)}/{len(teams)} selecciones con geo | agregación={args.aggregation}")

    rows = []
    for ii in range(len(idx)):
        for jj in range(ii + 1, len(idx)):
            i, j = idx[ii], idx[jj]
            a, b = resolved[i], resolved[j]
            rows.append({"a": a["country"], "b": b["country"], "sim": float(M[i, j]),
                         "dist": haversine(a, b), "colonial": colonial_link(a, b),
                         "lang": share_language(a, b)})

    sim = np.array([r["sim"] for r in rows])
    dist = np.array([r["dist"] for r in rows])
    col = np.array([1.0 if r["colonial"] else 0.0 for r in rows])

    pear = stats.pearsonr(sim, dist)
    spear = stats.spearmanr(sim, dist)
    simM = M[np.ix_(idx, idx)]
    distM = np.zeros_like(simM)
    for ii in range(len(idx)):
        for jj in range(len(idx)):
            if ii != jj:
                distM[ii, jj] = haversine(resolved[idx[ii]], resolved[idx[jj]])
    r_mantel, p_mantel = mantel(simM, distM, perms=args.perms)
    pc_dist = partial_corr(sim, dist, col)
    pc_col = partial_corr(sim, col, dist)

    print(f"[viz] Pearson(facial,dist)  r={pear.statistic:+.3f} p={pear.pvalue:.1e}")
    print(f"[viz] Spearman(facial,dist) rho={spear.statistic:+.3f} p={spear.pvalue:.1e}")
    print(f"[viz] Mantel r={r_mantel:+.3f} p={p_mantel:.4f} ({args.perms} perms)")
    print(f"[viz] parcial facial~dist | colonial = {pc_dist:+.3f}")
    print(f"[viz] parcial facial~colonial | dist  = {pc_col:+.3f}")
    print(f"[viz] pares: colonial={int(col.sum())} idioma={sum(r['lang'] for r in rows)} total={len(rows)}")

    fig, ax = plt.subplots(figsize=(11, 7.5))
    cats = [
        ("Sin lazo", [r for r in rows if not r["colonial"] and not r["lang"]], "#b0b0b0", 18, 0.5),
        ("Comparte idioma", [r for r in rows if r["lang"] and not r["colonial"]], "#2c7fb8", 34, 0.85),
        ("Lazo colonial", [r for r in rows if r["colonial"]], "#d7301f", 48, 0.95),
    ]
    for label, rs, c, sz, al in cats:
        if rs:
            ax.scatter([r["dist"] for r in rs], [r["sim"] for r in rs], s=sz, c=c,
                       alpha=al, edgecolors="none", label=f"{label} (n={len(rs)})")
    b1, b0 = np.polyfit(dist, sim, 1)
    xs = np.array([dist.min(), dist.max()])
    ax.plot(xs, b0 + b1 * xs, "k--", lw=1.3, label="tendencia (lineal)")
    ax.set_xlabel("Distancia entre capitales (km, great-circle)")
    ax.set_ylabel(f"Similitud facial equipo–equipo (coseno, {args.aggregation})")
    ax.set_title("Parecido facial entre selecciones vs. distancia y colonialidad — FIFA 48\n"
                 f"Mantel r={r_mantel:+.2f} (p={p_mantel:.3f}) · "
                 f"parcial facial~dist|colonial={pc_dist:+.2f} · facial~colonial|dist={pc_col:+.2f}")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, f"team_similarity_vs_geo_{args.tag}.png")
    fig.savefig(png, dpi=140)
    cpath = os.path.join(OUT_DIR, f"team_similarity_vs_geo_{args.tag}_pairs.csv")
    with open(cpath, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["equipo_a", "equipo_b", "similitud_facial", "distancia_km",
                    "lazo_colonial", "comparte_idioma"])
        for r in sorted(rows, key=lambda r: -r["sim"]):
            w.writerow([r["a"], r["b"], round(r["sim"], 4), round(r["dist"], 1),
                        "sí" if r["colonial"] else "", "sí" if r["lang"] else ""])
    print(f"[viz] PNG -> {png}\n[viz] PAIRS CSV -> {cpath}")


if __name__ == "__main__":
    sys.exit(main())
