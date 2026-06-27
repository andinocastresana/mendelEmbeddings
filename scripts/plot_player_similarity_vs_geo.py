#!/usr/bin/env python3
# =========================================
# ID: PHYLOFACE_PLAYER_SIM_VS_GEO
# VERSION: v0.1
# =========================================
# FILE: scripts/plot_player_similarity_vs_geo.py
#
# Versión a NIVEL JUGADOR del análisis parecido-vs-geo. Usa la matriz
# jugador-jugador del payload FIFA (1236x1236) y, para cada par CRUZA-EQUIPO,
# cruza la similitud facial con la distancia entre capitales de sus selecciones
# y el lazo colonial / idioma. Mucho más potente que el promedio equipo-equipo:
# un rostro individual no se aplana contra el plantel.
#
# Viz: densidad hexbin de todos los pares cruza-equipo + curvas de media por bin
# de distancia, separando lazo colonial vs sin lazo (muestra el "lift" colonial).
# Además dumpea los pares cruza-equipo más parecidos (con sus banderas).
#
# OJO estadístico: los pares jugador comparten estructura de bloque por equipo
# (no independientes) -> los p-values puntuales están inflados; el test honesto
# de significancia es el Mantel a nivel equipo (ver plot_team_similarity_vs_geo).
#
# Uso: conda run -n face-sim python scripts/plot_player_similarity_vs_geo.py

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facial", default=DEFAULT_FACIAL)
    ap.add_argument("--tag", default="fifa")
    ap.add_argument("--top", type=int, default=60, help="top pares cruza-equipo a dumpear.")
    args = ap.parse_args()

    print("[viz] cargando payload (42 MB)...")
    fac = json.load(open(args.facial, encoding="utf-8"))
    players = fac["players"]
    M = np.array(fac["player_similarity_matrix"], dtype=np.float32)
    n = len(players)
    resolve, _ = make_resolver()

    # Equipos únicos -> geo; índice de equipo por jugador.
    teams = fac["teams"]
    team_geo = {t: resolve(t) for t in teams}
    team_list = [t for t in teams if team_geo[t] is not None]
    tindex = {t: k for k, t in enumerate(team_list)}
    missing = [t for t in teams if team_geo[t] is None]
    if missing:
        print(f"[viz] equipos sin geo: {missing}")

    # Matrices equipo-equipo (distancia, colonial, idioma).
    T = len(team_list)
    distTT = np.zeros((T, T)); colTT = np.zeros((T, T), bool); langTT = np.zeros((T, T), bool)
    for i in range(T):
        for j in range(T):
            if i != j:
                a, b = team_geo[team_list[i]], team_geo[team_list[j]]
                distTT[i, j] = haversine(a, b)
                colTT[i, j] = colonial_link(a, b)
                langTT[i, j] = share_language(a, b)

    pteam = np.array([tindex.get(p["team"], -1) for p in players])
    valid = pteam >= 0

    # Pares cruza-equipo (triángulo superior), vectorizado.
    iu, ju = np.triu_indices(n, 1)
    keep = valid[iu] & valid[ju] & (pteam[iu] != pteam[ju])
    iu, ju = iu[keep], ju[keep]
    ti, tj = pteam[iu], pteam[ju]
    sim = M[iu, ju].astype(float)
    dist = distTT[ti, tj]
    col = colTT[ti, tj]
    lang = langTT[ti, tj]
    print(f"[viz] pares cruza-equipo: {len(sim):,} | colonial={int(col.sum()):,} idioma={int(lang.sum()):,}")

    pear = stats.pearsonr(sim, dist)
    spear = stats.spearmanr(sim, dist)
    pc_dist = partial_corr(sim, dist, col.astype(float))
    pc_col = partial_corr(sim, col.astype(float), dist)
    print(f"[viz] Pearson(facial,dist)  r={pear.statistic:+.3f} (p inflado por bloque)")
    print(f"[viz] Spearman(facial,dist) rho={spear.statistic:+.3f}")
    print(f"[viz] parcial facial~dist | colonial = {pc_dist:+.3f}")
    print(f"[viz] parcial facial~colonial | dist  = {pc_col:+.3f}")
    print(f"[viz] media sim: colonial={sim[col].mean():.4f} vs sin-lazo={sim[~col].mean():.4f}")

    # --- Viz: hexbin + curvas de media por bin de distancia ---
    fig, ax = plt.subplots(figsize=(11, 7.5))
    hb = ax.hexbin(dist, sim, gridsize=55, cmap="Greys", bins="log", mincnt=1)
    edges = np.linspace(0, dist.max(), 21)
    centers = (edges[:-1] + edges[1:]) / 2
    for mask, c, lab in [(col, "#d7301f", "Lazo colonial"),
                         (~col & lang, "#2c7fb8", "Comparte idioma (no colonial)"),
                         (~col & ~lang, "#444444", "Sin lazo")]:
        means = []
        for k in range(len(edges) - 1):
            m = mask & (dist >= edges[k]) & (dist < edges[k + 1])
            means.append(sim[m].mean() if m.sum() >= 20 else np.nan)
        ax.plot(centers, means, "-o", color=c, ms=4, lw=2, label=f"{lab} (media/bin)")
    cb = fig.colorbar(hb, ax=ax); cb.set_label("densidad de pares (log)")
    ax.set_xlabel("Distancia entre capitales de las selecciones (km)")
    ax.set_ylabel("Similitud facial jugador–jugador (coseno)")
    ax.set_title("Parecido facial a NIVEL JUGADOR vs. distancia y colonialidad — FIFA 48\n"
                 f"parcial facial~dist|colonial={pc_dist:+.2f} · facial~colonial|dist={pc_col:+.2f} · "
                 f"media colonial {sim[col].mean():.3f} vs sin-lazo {sim[~col].mean():.3f}")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, f"player_similarity_vs_geo_{args.tag}.png")
    fig.savefig(png, dpi=140)
    print(f"[viz] PNG -> {png}")

    # --- Top pares cruza-equipo más parecidos ---
    order = np.argsort(-sim)[:args.top]
    cpath = os.path.join(OUT_DIR, f"player_similarity_top_crossteam_{args.tag}.csv")
    with open(cpath, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["jugador_a", "equipo_a", "jugador_b", "equipo_b", "similitud",
                    "distancia_km", "lazo_colonial", "comparte_idioma"])
        for k in order:
            pa, pb = players[iu[k]], players[ju[k]]
            w.writerow([pa["name"], pa["team"], pb["name"], pb["team"],
                        round(float(sim[k]), 4), round(float(dist[k]), 1),
                        "sí" if col[k] else "", "sí" if lang[k] else ""])
    print(f"[viz] TOP pares -> {cpath}")
    for k in order[:6]:
        pa, pb = players[iu[k]], players[ju[k]]
        print(f"   {pa['name']} ({pa['team']}) ↔ {pb['name']} ({pb['team']}): "
              f"{sim[k]:.3f}  {'COLONIAL' if col[k] else ''}{' idioma' if lang[k] else ''}")


if __name__ == "__main__":
    sys.exit(main())
