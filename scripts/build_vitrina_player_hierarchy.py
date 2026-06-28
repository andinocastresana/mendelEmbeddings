#!/usr/bin/env python3
# =========================================
# ID: PHYLOFACE_VITRINA_PLAYER_HIERARCHY
# VERSION: v0.1
# =========================================
# FILE: scripts/build_vitrina_player_hierarchy.py
#
# Precalcula ordenes jerarquicos jugador-jugador para visualizar heatmaps grandes
# en canvas dentro del viewer local de vitrina.

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data/output/teams/vitrina_fifa_northamerica2026_similarity_pilot.json"
)
DEFAULT_CLUSTER_INPUT = (
    PROJECT_ROOT
    / "data/output/teams/vitrina_fifa_northamerica2026_cluster_exploration.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data/output/teams/vitrina_fifa_northamerica2026_player_hierarchy.json"
)


def rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def clean_float_matrix(matrix: np.ndarray) -> list[list[float | None]]:
    rows: list[list[float | None]] = []
    for row in matrix:
        rows.append([None if not np.isfinite(value) else float(value) for value in row])
    return rows


def hierarchy_order(similarity: np.ndarray, indices: list[int], method: str) -> tuple[list[int], dict[str, Any]]:
    sub = similarity[np.ix_(indices, indices)]
    distance = np.clip(1.0 - sub, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    z = linkage(condensed, method=method, optimal_ordering=True)
    order_local = leaves_list(z).astype(int).tolist()
    ordered = [indices[i] for i in order_local]
    return ordered, {
        "linkage": method,
        "merge_distance_min": float(np.min(z[:, 2])) if z.size else None,
        "merge_distance_max": float(np.max(z[:, 2])) if z.size else None,
    }


def player_payload(player: dict[str, Any]) -> dict[str, Any]:
    return {
        "idx": player["idx"],
        "name": player["name"],
        "team": player["team"],
        "position": player.get("position"),
        "local_image": player.get("local_image"),
    }


def all_players_subset(players: list[dict[str, Any]], max_players: int) -> list[int]:
    return [player["idx"] for player in players[:max_players]]


def balanced_by_team_subset(players: list[dict[str, Any]], max_per_team: int) -> list[int]:
    selected = []
    counts: Counter[str] = Counter()
    for player in players:
        if counts[player["team"]] >= max_per_team:
            continue
        selected.append(player["idx"])
        counts[player["team"]] += 1
    return selected


def clusters_from_payload(cluster_payload: dict[str, Any], method_index: int) -> list[dict[str, Any]]:
    methods = cluster_payload.get("methods", [])
    if not methods:
        return []
    method = methods[min(method_index, len(methods) - 1)]
    return method.get("summary", {}).get("clusters", [])


def cluster_subset(
    clusters: list[dict[str, Any]],
    *,
    cluster_id: int,
    max_players: int,
) -> list[int]:
    for cluster in clusters:
        if int(cluster["cluster"]) == cluster_id:
            return [player["idx"] for player in cluster.get("players", [])[:max_players]]
    return []


def build_subset(
    name: str,
    label: str,
    indices: list[int],
    *,
    players: list[dict[str, Any]],
    similarity: np.ndarray,
    linkage_method: str,
) -> dict[str, Any]:
    indices = list(dict.fromkeys(indices))
    ordered, hierarchy = hierarchy_order(similarity, indices, linkage_method)
    player_by_idx = {player["idx"]: player for player in players}
    ordered_players = [player_payload(player_by_idx[idx]) for idx in ordered]
    sub = similarity[np.ix_(ordered, ordered)]
    teams = Counter(player_by_idx[idx]["team"] for idx in ordered)
    return {
        "name": name,
        "label": label,
        "players_count": len(ordered),
        "teams_count": len(teams),
        "team_counts": dict(teams.most_common()),
        "order": ordered,
        "players": ordered_players,
        "similarity_matrix": clean_float_matrix(sub),
        "hierarchy": hierarchy,
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    source = json.loads(args.input.read_text(encoding="utf-8"))
    players = source["players"]
    similarity = np.asarray(source["player_similarity_matrix"], dtype=np.float32)
    cluster_payload = json.loads(args.cluster_input.read_text(encoding="utf-8"))
    clusters = clusters_from_payload(cluster_payload, args.cluster_method_index)

    subsets = [
        build_subset(
            "balanced_teams",
            f"Balanceado por seleccion · max {args.max_per_team} jugadores",
            balanced_by_team_subset(players, args.max_per_team),
            players=players,
            similarity=similarity,
            linkage_method=args.linkage,
        ),
        build_subset(
            "first_players",
            f"Primeros {args.max_players} jugadores del payload",
            all_players_subset(players, args.max_players),
            players=players,
            similarity=similarity,
            linkage_method=args.linkage,
        ),
    ]
    for cluster in clusters[: args.cluster_subsets]:
        cluster_id = int(cluster["cluster"])
        indices = cluster_subset(clusters, cluster_id=cluster_id, max_players=args.max_players)
        if len(indices) < 2:
            continue
        subsets.append(
            build_subset(
                f"cluster_{cluster_id}",
                f"Cluster {cluster_id} · {len(indices)} jugadores",
                indices,
                players=players,
                similarity=similarity,
                linkage_method=args.linkage,
            ),
        )

    return {
        "schema": "phyloface-vitrina-player-hierarchy-v0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_similarity_payload": rel_path(args.input),
        "source_cluster_payload": rel_path(args.cluster_input),
        "cluster_method_index": args.cluster_method_index,
        "embedding_model": source.get("embedding_model"),
        "metric": "cosine",
        "distance": "1 - cosine",
        "linkage": args.linkage,
        "subsets": subsets,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precalcula jerarquia jugador-jugador para la vitrina.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--cluster-input", type=Path, default=DEFAULT_CLUSTER_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--linkage", choices=["average", "complete", "single"], default="average")
    parser.add_argument("--max-players", type=int, default=360)
    parser.add_argument("--max-per-team", type=int, default=6)
    parser.add_argument("--cluster-method-index", type=int, default=0)
    parser.add_argument("--cluster-subsets", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.input = args.input.resolve()
    args.cluster_input = args.cluster_input.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(args)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Jerarquia escrita: {args.output}")
    for subset in payload["subsets"]:
        print(f"{subset['name']}: {subset['players_count']} jugadores · {subset['teams_count']} selecciones")


if __name__ == "__main__":
    main()
