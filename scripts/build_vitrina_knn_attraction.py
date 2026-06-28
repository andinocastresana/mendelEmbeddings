#!/usr/bin/env python3
# =========================================
# ID: PHYLOFACE_VITRINA_KNN_ATTRACTION
# VERSION: v0.1
# =========================================
# FILE: scripts/build_vitrina_knn_attraction.py
#
# Construye una vista de "atraccion externa" por kNN sobre la matriz de similitud
# jugador-jugador del payload de vitrina. Vecinos se calculan en la similitud
# original, no sobre reducciones 2D.

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data/output/teams/vitrina_fifa_northamerica2026_similarity_pilot.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data/output/teams/vitrina_fifa_northamerica2026_knn_attraction.json"
)


def rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def top_external_neighbors(
    matrix: np.ndarray,
    players: list[dict[str, Any]],
    idx: int,
    k: int,
) -> list[tuple[int, float]]:
    source_team = players[idx]["team"]
    scores = matrix[idx].astype(np.float32).copy()
    scores[idx] = -np.inf
    for j, player in enumerate(players):
        if player["team"] == source_team:
            scores[j] = -np.inf
    finite = np.isfinite(scores)
    if not finite.any():
        return []
    k = min(k, int(finite.sum()))
    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return [(int(j), float(scores[j])) for j in top_idx]


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    source = json.loads(args.input.read_text(encoding="utf-8"))
    players = source["players"]
    teams = source["teams"]
    matrix = np.asarray(source["player_similarity_matrix"], dtype=np.float32)

    outgoing: dict[str, dict[str, dict[str, float]]] = {
        team: defaultdict(lambda: {"count": 0, "score_sum": 0.0})
        for team in teams
    }
    player_neighbors = []

    for idx, player in enumerate(players):
        neighbors = top_external_neighbors(matrix, players, idx, args.k)
        packed_neighbors = []
        for neighbor_idx, score in neighbors:
            neighbor = players[neighbor_idx]
            dest_team = neighbor["team"]
            outgoing[player["team"]][dest_team]["count"] += 1
            outgoing[player["team"]][dest_team]["score_sum"] += score
            packed_neighbors.append(
                {
                    "idx": neighbor_idx,
                    "name": neighbor["name"],
                    "team": dest_team,
                    "position": neighbor.get("position"),
                    "score": score,
                },
            )
        player_neighbors.append(
            {
                "idx": idx,
                "name": player["name"],
                "team": player["team"],
                "position": player.get("position"),
                "neighbors": packed_neighbors,
            },
        )

    team_rows = []
    matrix_count = np.zeros((len(teams), len(teams)), dtype=np.float32)
    matrix_score = np.zeros((len(teams), len(teams)), dtype=np.float32)
    team_to_idx = {team: idx for idx, team in enumerate(teams)}
    for source_team in teams:
        destinations = []
        total_count = sum(item["count"] for item in outgoing[source_team].values())
        for dest_team, values in outgoing[source_team].items():
            count = int(values["count"])
            score_sum = float(values["score_sum"])
            share = count / total_count if total_count else 0.0
            mean_score = score_sum / count if count else None
            destinations.append(
                {
                    "team": dest_team,
                    "count": count,
                    "share": share,
                    "mean_score": mean_score,
                    "score_sum": score_sum,
                },
            )
            i = team_to_idx[source_team]
            j = team_to_idx[dest_team]
            matrix_count[i, j] = share
            matrix_score[i, j] = 0.0 if mean_score is None else mean_score
        destinations.sort(key=lambda item: (item["count"], item["mean_score"] or -1), reverse=True)
        team_rows.append(
            {
                "team": source_team,
                "players_count": source["team_player_counts"].get(source_team, 0),
                "neighbors_count": int(total_count),
                "top_destinations": destinations[: args.top_destinations],
                "destinations": destinations,
            },
        )

    return {
        "schema": "phyloface-vitrina-knn-attraction-v0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_similarity_payload": rel_path(args.input),
        "embedding_model": source.get("embedding_model"),
        "k": args.k,
        "players_count": len(players),
        "teams_count": len(teams),
        "teams": teams,
        "team_player_counts": source["team_player_counts"],
        "team_attraction": team_rows,
        "team_attraction_share_matrix": matrix_count.astype(float).tolist(),
        "team_attraction_mean_score_matrix": matrix_score.astype(float).tolist(),
        "player_neighbors": player_neighbors if args.include_player_neighbors else [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye atraccion externa kNN por seleccion.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--top-destinations", type=int, default=8)
    parser.add_argument("--include-player-neighbors", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(args)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Atraccion kNN escrita: {args.output}")
    print(
        f"k={payload['k']} · jugadores={payload['players_count']} · "
        f"selecciones={payload['teams_count']}",
    )


if __name__ == "__main__":
    main()
