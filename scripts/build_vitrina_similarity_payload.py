#!/usr/bin/env python3
# =========================================
# ID: PHYLOFACE_VITRINA_SIMILARITY_PAYLOAD
# VERSION: v0.1
# =========================================
# FILE: scripts/build_vitrina_similarity_payload.py
#
# Construye un JSON estatico para la vitrina a partir del QC de retratos:
# jugadores aceptados, matriz coseno jugador-jugador, matriz seleccion-seleccion
# y rankings de pares mas parecidos.

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data/output/teams/manifest_transfermarkt_northamerica2026_headshots_qc.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data/output/teams/vitrina_transfermarkt_northamerica2026_similarity_pilot.json"
)


def rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def accepted_players(qc: dict[str, Any]) -> list[dict[str, Any]]:
    players = []
    for row in qc.get("players", []):
        if row.get("qc_status") != "accepted" or "embedding" not in row:
            continue
        players.append(row)
    return players


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def player_payload(row: dict[str, Any], idx: int) -> dict[str, Any]:
    return {
        "idx": idx,
        "team": row.get("team"),
        "number": row.get("number"),
        "position": row.get("position"),
        "name": row.get("name"),
        "local_image": row.get("local_image"),
        "metrics": row.get("metrics"),
    }


def team_indices(players: list[dict[str, Any]]) -> dict[str, list[int]]:
    teams: dict[str, list[int]] = {}
    for idx, player in enumerate(players):
        teams.setdefault(player["team"], []).append(idx)
    return dict(sorted(teams.items()))


def pair_payload(
    players: list[dict[str, Any]],
    matrix: np.ndarray,
    *,
    include_same_team: bool,
    limit: int,
) -> list[dict[str, Any]]:
    pairs = []
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            if not include_same_team and players[i]["team"] == players[j]["team"]:
                continue
            pairs.append((float(matrix[i, j]), i, j))
    pairs.sort(reverse=True)
    return [
        {
            "score": score,
            "a_idx": i,
            "b_idx": j,
            "a_name": players[i]["name"],
            "b_name": players[j]["name"],
            "a_team": players[i]["team"],
            "b_team": players[j]["team"],
        }
        for score, i, j in pairs[:limit]
    ]


def team_similarity(players: list[dict[str, Any]], matrix: np.ndarray) -> dict[str, Any]:
    indices_by_team = team_indices(players)
    teams = list(indices_by_team)
    team_mean_matrix = np.zeros((len(teams), len(teams)), dtype=np.float32)
    team_median_matrix = np.zeros((len(teams), len(teams)), dtype=np.float32)
    team_top3_mean_matrix = np.zeros((len(teams), len(teams)), dtype=np.float32)
    team_top5_mean_matrix = np.zeros((len(teams), len(teams)), dtype=np.float32)
    intra_stats = {}

    for a, team_a in enumerate(teams):
        idx_a = indices_by_team[team_a]
        for b, team_b in enumerate(teams):
            idx_b = indices_by_team[team_b]
            sub = matrix[np.ix_(idx_a, idx_b)]
            if team_a == team_b:
                values = sub[np.triu_indices_from(sub, k=1)]
                if values.size == 0:
                    team_mean_matrix[a, b] = float("nan")
                    team_median_matrix[a, b] = float("nan")
                    team_top3_mean_matrix[a, b] = float("nan")
                    team_top5_mean_matrix[a, b] = float("nan")
                else:
                    team_mean_matrix[a, b] = float(values.mean())
                    team_median_matrix[a, b] = float(np.median(values))
                    team_top3_mean_matrix[a, b] = topk_mean(values, 3)
                    team_top5_mean_matrix[a, b] = topk_mean(values, 5)
            else:
                values = sub.ravel()
                team_mean_matrix[a, b] = float(values.mean())
                team_median_matrix[a, b] = float(np.median(values))
                team_top3_mean_matrix[a, b] = topk_mean(values, 3)
                team_top5_mean_matrix[a, b] = topk_mean(values, 5)

        intra_values = matrix[np.ix_(idx_a, idx_a)]
        intra_values = intra_values[np.triu_indices_from(intra_values, k=1)]
        intra_stats[team_a] = stats_payload(intra_values)

    mean_payload = clean_float_matrix(team_mean_matrix)
    median_payload = clean_float_matrix(team_median_matrix)
    top3_mean_payload = clean_float_matrix(team_top3_mean_matrix)
    top5_mean_payload = clean_float_matrix(team_top5_mean_matrix)
    return {
        "teams": teams,
        "team_player_counts": {team: len(indices) for team, indices in indices_by_team.items()},
        "team_similarity_matrix": mean_payload,
        "team_similarity_matrices": {
            "mean": mean_payload,
            "median": median_payload,
            "top3_mean": top3_mean_payload,
            "top5_mean": top5_mean_payload,
        },
        "team_similarity_aggregation": "mean",
        "intra_team_stats": intra_stats,
    }


def topk_mean(values: np.ndarray, k: int) -> float:
    flat = np.asarray(values, dtype=np.float32).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return float("nan")
    k = min(k, flat.size)
    top = np.partition(flat, flat.size - k)[-k:]
    return float(top.mean())


def stats_payload(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def clean_float_matrix(matrix: np.ndarray) -> list[list[float | None]]:
    rows: list[list[float | None]] = []
    for row in matrix:
        rows.append([None if not np.isfinite(value) else float(value) for value in row])
    return rows


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    qc = json.loads(args.input.read_text(encoding="utf-8"))
    rows = accepted_players(qc)
    if not rows:
        raise ValueError("El QC no contiene jugadores aceptados con embedding.")

    embeddings = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
    normalized = l2_normalize(embeddings)
    matrix = normalized @ normalized.T

    players = [player_payload(row, idx) for idx, row in enumerate(rows)]
    team_payload = team_similarity(players, matrix)

    return {
        "schema": "phyloface-vitrina-similarity-payload-v0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_qc_manifest": rel_path(args.input),
        "embedding_model": qc.get("model_name"),
        "players_count": len(players),
        "teams_count": len(team_payload["teams"]),
        "players": players,
        "player_similarity_matrix": clean_float_matrix(matrix),
        **team_payload,
        "top_pairs_cross_team": pair_payload(
            players,
            matrix,
            include_same_team=False,
            limit=args.top_pairs,
        ),
        "top_pairs_all": pair_payload(
            players,
            matrix,
            include_same_team=True,
            limit=args.top_pairs,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye payload estatico de similitudes para la vitrina.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Manifiesto QC.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON de salida.")
    parser.add_argument("--top-pairs", type=int, default=100, help="Cantidad de pares destacados.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(args)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Payload escrito: {args.output}")
    print(f"Jugadores: {payload['players_count']} · selecciones: {payload['teams_count']}")


if __name__ == "__main__":
    main()
