#!/usr/bin/env python3
# =========================================
# ID: PHYLOFACE_VITRINA_CLUSTER_EXPLORATION
# VERSION: v0.1
# =========================================
# FILE: scripts/build_vitrina_cluster_exploration.py
#
# Explora agrupamientos jugador-jugador para la vitrina sin depender de una
# proyeccion 2D. Compara comunidades kNN, clustering aglomerativo, spectral y un
# grafo multiplex facial+seleccion.

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data/output/teams/vitrina_fifa_northamerica2026_similarity_pilot.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data/output/teams/vitrina_fifa_northamerica2026_cluster_exploration.json"
)


def rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def finite_offdiag(matrix: np.ndarray) -> np.ndarray:
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    values = matrix[mask]
    return values[np.isfinite(values)]


def normalized_affinity(matrix: np.ndarray) -> np.ndarray:
    values = finite_offdiag(matrix)
    lo = float(values.min())
    hi = float(values.max())
    span = hi - lo if hi > lo else 1.0
    affinity = (matrix - lo) / span
    affinity = np.clip(affinity, 0.0, 1.0)
    np.fill_diagonal(affinity, 0.0)
    return affinity.astype(np.float32)


def build_knn_graph(
    affinity: np.ndarray,
    *,
    k: int,
    players: list[dict[str, Any]],
    external_only: bool = False,
) -> nx.Graph:
    graph = nx.Graph()
    for idx, player in enumerate(players):
        graph.add_node(idx, team=player["team"], name=player["name"])

    for i, player in enumerate(players):
        scores = affinity[i].copy()
        scores[i] = -np.inf
        if external_only:
            for j, other in enumerate(players):
                if other["team"] == player["team"]:
                    scores[j] = -np.inf
        finite = np.isfinite(scores)
        if not finite.any():
            continue
        local_k = min(k, int(finite.sum()))
        top_idx = np.argpartition(scores, -local_k)[-local_k:]
        for j in top_idx:
            weight = float(scores[j])
            if not math.isfinite(weight) or weight <= 0:
                continue
            if graph.has_edge(i, int(j)):
                graph[i][int(j)]["weight"] = max(graph[i][int(j)]["weight"], weight)
            else:
                graph.add_edge(i, int(j), weight=weight)
    return graph


def greedy_communities(graph: nx.Graph) -> list[int]:
    communities = nx.algorithms.community.greedy_modularity_communities(
        graph,
        weight="weight",
    )
    communities = sorted(communities, key=lambda group: (-len(group), min(group)))
    labels = [-1] * graph.number_of_nodes()
    for cluster_id, group in enumerate(communities):
        for idx in group:
            labels[int(idx)] = cluster_id
    return labels


def agglomerative_labels(distance: np.ndarray, n_clusters: int, linkage: str) -> list[int]:
    from sklearn.cluster import AgglomerativeClustering

    kwargs: dict[str, Any] = {
        "n_clusters": n_clusters,
        "linkage": linkage,
    }
    try:
        model = AgglomerativeClustering(metric="precomputed", **kwargs)
    except TypeError:
        model = AgglomerativeClustering(affinity="precomputed", **kwargs)
    return [int(value) for value in model.fit_predict(distance)]


def spectral_labels(affinity: np.ndarray, n_clusters: int, random_state: int) -> list[int]:
    from sklearn.cluster import SpectralClustering

    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=random_state,
    )
    return [int(value) for value in model.fit_predict(affinity)]


def entropy(counter: Counter[str | int]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    result = 0.0
    for count in counter.values():
        if count:
            p = count / total
            result -= p * math.log(p)
    return result


def normalized_entropy(counter: Counter[str | int]) -> float:
    if len(counter) <= 1:
        return 0.0
    return entropy(counter) / math.log(len(counter))


def summarize_partition(
    labels: list[int],
    *,
    players: list[dict[str, Any]],
    limit_examples: int,
) -> dict[str, Any]:
    clusters: dict[int, list[int]] = defaultdict(list)
    teams: dict[str, list[int]] = defaultdict(list)
    for idx, (label, player) in enumerate(zip(labels, players, strict=True)):
        clusters[int(label)].append(idx)
        teams[player["team"]].append(idx)

    sizes = np.asarray([len(indices) for indices in clusters.values()], dtype=np.float32)
    cluster_rows = []
    weighted_cluster_entropy = 0.0
    external_peer_share_sum = 0.0
    external_peer_share_n = 0

    for label, indices in clusters.items():
        team_counts = Counter(players[idx]["team"] for idx in indices)
        weighted_cluster_entropy += len(indices) * normalized_entropy(team_counts)
        for idx in indices:
            peers = [other for other in indices if other != idx]
            if not peers:
                continue
            external = sum(1 for other in peers if players[other]["team"] != players[idx]["team"])
            external_peer_share_sum += external / len(peers)
            external_peer_share_n += 1
        cluster_rows.append(
            {
                "cluster": int(label),
                "size": len(indices),
                "teams_count": len(team_counts),
                "team_entropy_norm": normalized_entropy(team_counts),
                "top_teams": [
                    {"team": team, "count": count}
                    for team, count in team_counts.most_common()
                ],
                "players": [
                    {
                        "idx": idx,
                        "name": players[idx]["name"],
                        "team": players[idx]["team"],
                        "position": players[idx].get("position"),
                        "local_image": players[idx].get("local_image"),
                    }
                    for idx in indices
                ],
            },
        )

    team_rows = []
    weighted_team_entropy = 0.0
    for team, indices in teams.items():
        cluster_counts = Counter(labels[idx] for idx in indices)
        weighted_team_entropy += len(indices) * normalized_entropy(cluster_counts)
        team_rows.append(
            {
                "team": team,
                "players_count": len(indices),
                "clusters_count": len(cluster_counts),
                "cluster_entropy_norm": normalized_entropy(cluster_counts),
                "top_clusters": [
                    {"cluster": int(cluster), "count": count}
                    for cluster, count in cluster_counts.most_common(8)
                ],
            },
        )

    cluster_rows.sort(key=lambda item: (-item["size"], item["cluster"]))
    team_rows.sort(key=lambda item: (-item["cluster_entropy_norm"], item["team"]))
    return {
        "clusters_count": len(clusters),
        "size": {
            "min": int(sizes.min()),
            "median": float(np.median(sizes)),
            "mean": float(sizes.mean()),
            "max": int(sizes.max()),
            "singletons": int((sizes == 1).sum()),
        },
        "weighted_cluster_team_entropy_norm": weighted_cluster_entropy / len(players),
        "weighted_team_cluster_entropy_norm": weighted_team_entropy / len(players),
        "mean_external_peer_share": (
            external_peer_share_sum / external_peer_share_n
            if external_peer_share_n
            else None
        ),
        "clusters": cluster_rows,
        "largest_clusters": cluster_rows[:limit_examples],
        "most_fragmented_teams": team_rows[:limit_examples],
    }


def run_exploration(args: argparse.Namespace) -> dict[str, Any]:
    source = json.loads(args.input.read_text(encoding="utf-8"))
    players = source["players"]
    raw_sim = np.asarray(source["player_similarity_matrix"], dtype=np.float32)
    affinity = normalized_affinity(raw_sim)
    distance = np.clip(1.0 - raw_sim, 0.0, 2.0).astype(np.float32)
    np.fill_diagonal(distance, 0.0)

    methods = []

    k_values = args.k_values or [args.k]

    for k in k_values:
        for external_only in [False, True]:
            graph = build_knn_graph(
                affinity,
                k=k,
                players=players,
                external_only=external_only,
            )
            labels = greedy_communities(graph)
            methods.append(
                {
                    "method": "knn_greedy_modularity",
                    "params": {
                        "k": k,
                        "external_only": external_only,
                        "nodes": graph.number_of_nodes(),
                        "edges": graph.number_of_edges(),
                    },
                    "summary": summarize_partition(
                        labels,
                        players=players,
                        limit_examples=args.examples,
                    ),
                },
            )

    for n_clusters in args.cluster_counts:
        for linkage in args.linkages:
            labels = agglomerative_labels(distance, n_clusters, linkage)
            methods.append(
                {
                    "method": "agglomerative",
                    "params": {"n_clusters": n_clusters, "linkage": linkage},
                    "summary": summarize_partition(
                        labels,
                        players=players,
                        limit_examples=args.examples,
                    ),
                },
            )

        for k in k_values:
            graph = build_knn_graph(
                affinity,
                k=k,
                players=players,
                external_only=False,
            )
            graph_affinity = nx.to_numpy_array(graph, nodelist=range(len(players)), weight="weight")
            labels = spectral_labels(graph_affinity, n_clusters, args.random_state)
            methods.append(
                {
                    "method": "spectral_knn",
                    "params": {
                        "n_clusters": n_clusters,
                        "k": k,
                        "random_state": args.random_state,
                    },
                    "summary": summarize_partition(
                        labels,
                        players=players,
                        limit_examples=args.examples,
                    ),
                },
            )

    same_team = np.zeros_like(affinity)
    teams = [player["team"] for player in players]
    for i, team_i in enumerate(teams):
        for j, team_j in enumerate(teams):
            if i != j and team_i == team_j:
                same_team[i, j] = 1.0
    for k in k_values:
        for beta in args.multiplex_betas:
            multiplex = affinity + beta * same_team
            np.fill_diagonal(multiplex, 0.0)
            graph = build_knn_graph(
                multiplex,
                k=k,
                players=players,
                external_only=False,
            )
            labels = greedy_communities(graph)
            methods.append(
                {
                    "method": "multiplex_knn_greedy_modularity",
                    "params": {
                        "k": k,
                        "beta_same_team": beta,
                        "nodes": graph.number_of_nodes(),
                        "edges": graph.number_of_edges(),
                    },
                    "summary": summarize_partition(
                        labels,
                        players=players,
                        limit_examples=args.examples,
                    ),
                },
            )

    return {
        "schema": "phyloface-vitrina-cluster-exploration-v0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_similarity_payload": rel_path(args.input),
        "embedding_model": source.get("embedding_model"),
        "players_count": len(players),
        "teams_count": len(source["teams"]),
        "similarity": {
            "offdiag_min": float(finite_offdiag(raw_sim).min()),
            "offdiag_max": float(finite_offdiag(raw_sim).max()),
            "offdiag_mean": float(finite_offdiag(raw_sim).mean()),
        },
        "methods": methods,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explora clustering/grafos para la vitrina FIFA.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--k-values", type=parse_int_list, default=None)
    parser.add_argument("--cluster-counts", type=parse_int_list, default=parse_int_list("24,48,96"))
    parser.add_argument("--linkages", type=lambda value: [x.strip() for x in value.split(",") if x.strip()], default=["average"])
    parser.add_argument("--multiplex-betas", type=parse_float_list, default=parse_float_list("0.05,0.10,0.20"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--examples", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = run_exploration(args)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Exploracion escrita: {args.output}")
    print(
        f"jugadores={payload['players_count']} · selecciones={payload['teams_count']} · "
        f"metodos={len(payload['methods'])}",
    )
    for item in payload["methods"]:
        summary = item["summary"]
        print(
            f"{item['method']} {item['params']} -> "
            f"clusters={summary['clusters_count']} · "
            f"size_med={summary['size']['median']:.1f} · "
            f"team_entropy={summary['weighted_cluster_team_entropy_norm']:.3f} · "
            f"fragmentacion={summary['weighted_team_cluster_entropy_norm']:.3f}",
        )


if __name__ == "__main__":
    main()
