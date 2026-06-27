#!/usr/bin/env python3
# =========================================
# ID: PHYLOFACE_TEAM_SIM_MAP
# VERSION: v0.1
# =========================================
# FILE: scripts/plot_team_similarity_map.py
#
# Mapa mundi con las capitales de las 48 selecciones como nodos y aristas entre
# los pares facialmente más parecidos (top-k de la matriz equipo-equipo FIFA).
# Las aristas que CRUZAN regiones (rojo) resaltan los "dobles" trans-oceánicos,
# típicamente lazos coloniales/migratorios; las intra-región van en azul tenue.
#
# Basemap: Natural Earth 110m (cacheado en data/input/geo/; fallback sin costas).
#
# Uso: conda run -n face-sim python scripts/plot_team_similarity_map.py --topk 50

import argparse
import json
import os
import sys

import numpy as np
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geo_team_resolve import make_resolver, colonial_link

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_FACIAL = os.path.join(REPO_ROOT, "data", "output", "teams",
                              "vitrina_fifa_northamerica2026_similarity_pilot.json")
OUT_DIR = os.path.join(REPO_ROOT, "data", "output", "geo")
RAW_DIR = os.path.join(REPO_ROOT, "data", "input", "geo")
BASEMAP_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
               "master/geojson/ne_110m_admin_0_countries.geojson")
REGION_COLORS = {"Americas": "#4e79a7", "Europe": "#59a14f", "Africa": "#e15759",
                 "Asia": "#f28e2b", "Oceania": "#b07aa1", None: "#888888"}


def load_basemap():
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, "ne_110m_admin_0_countries.geojson")
    if not os.path.exists(path):
        try:
            r = requests.get(BASEMAP_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
            r.raise_for_status()
            open(path, "wb").write(r.content)
        except Exception as e:
            print(f"[map] sin basemap ({e}); sigo sin costas.")
            return None
    return json.load(open(path, encoding="utf-8"))


def draw_basemap(ax, gj):
    if not gj:
        return
    for feat in gj["features"]:
        geom = feat["geometry"]
        if not geom:
            continue
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            for ring in poly:
                xy = np.array(ring)
                ax.plot(xy[:, 0], xy[:, 1], color="#d9d9d9", lw=0.4, zorder=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facial", default=DEFAULT_FACIAL)
    ap.add_argument("--aggregation", default="mean",
                    choices=["mean", "median", "top3_mean", "top5_mean"])
    ap.add_argument("--topk", type=int, default=50, help="nº de aristas (pares más parecidos).")
    ap.add_argument("--tag", default="fifa")
    args = ap.parse_args()

    fac = json.load(open(args.facial, encoding="utf-8"))
    teams = fac["teams"]
    mats = fac.get("team_similarity_matrices") or {}
    M = np.array(mats.get(args.aggregation) or fac["team_similarity_matrix"], dtype=float)
    resolve, _ = make_resolver()

    geo = [resolve(t) for t in teams]
    idx = [i for i, g in enumerate(geo) if g is not None]

    # Top-k pares por similitud (triángulo superior, solo equipos con geo).
    pairs = [(i, j, M[i, j]) for a, i in enumerate(idx) for j in idx[a + 1:]]
    pairs.sort(key=lambda p: -p[2])
    top = pairs[:args.topk]

    fig, ax = plt.subplots(figsize=(15, 8))
    draw_basemap(ax, load_basemap())

    n_cross = 0
    for i, j, s in top:
        a, b = geo[i], geo[j]
        cross = a["region"] != b["region"]
        n_cross += cross
        lw = 0.6 + 6.0 * (s - top[-1][2]) / (top[0][2] - top[-1][2] + 1e-9)
        ax.plot([a["lon"], b["lon"]], [a["lat"], b["lat"]],
                color=("#d7301f" if cross else "#9ecae1"),
                lw=lw, alpha=0.55 if cross else 0.4,
                zorder=3 if cross else 2)

    for i in idx:
        g = geo[i]
        ax.scatter(g["lon"], g["lat"], s=46, zorder=4,
                   c=REGION_COLORS.get(g["region"], "#888888"), edgecolors="white", linewidths=0.6)
        ax.annotate(teams[i], (g["lon"], g["lat"]), fontsize=6.2, zorder=5,
                    xytext=(3, 3), textcoords="offset points")

    ax.set_xlim(-180, 180); ax.set_ylim(-60, 85)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    handles = [plt.Line2D([], [], color="#d7301f", lw=3, label=f"cruza región (n={n_cross})"),
               plt.Line2D([], [], color="#9ecae1", lw=3, label=f"intra región (n={len(top)-n_cross})")]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9)
    ax.set_title(f"Selecciones más parecidas facialmente — top {args.topk} aristas (FIFA 48, {args.aggregation})\n"
                 "rojo = parecido que cruza regiones (puentes coloniales/migratorios)")
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, f"team_similarity_map_{args.tag}.png")
    fig.savefig(png, dpi=150)
    print(f"[map] top{args.topk}: {n_cross} cruza-región / {len(top)-n_cross} intra")
    print("[map] aristas cruza-región más fuertes:")
    for i, j, s in [p for p in top if geo[p[0]]["region"] != geo[p[1]]["region"]][:8]:
        print(f"   {teams[i]} ↔ {teams[j]}: {s:.3f}  "
              f"{'COLONIAL' if colonial_link(geo[i], geo[j]) else ''}")
    print(f"[map] PNG -> {png}")


if __name__ == "__main__":
    sys.exit(main())
