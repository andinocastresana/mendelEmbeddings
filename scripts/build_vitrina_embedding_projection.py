#!/usr/bin/env python3
# =========================================
# ID: PHYLOFACE_VITRINA_EMBEDDING_PROJECTION
# VERSION: v0.1
# =========================================
# FILE: scripts/build_vitrina_embedding_projection.py
#
# Proyecta embeddings full-face de jugadores aceptados por QC a 2D para la
# vitrina. La salida es un JSON pequeño para visualización local/client-side.

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
    / "data/output/teams/manifest_fifa_northamerica2026_official_qc.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data/output/teams/vitrina_fifa_northamerica2026_projection_tsne.json"
)


def rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def accepted_players(qc: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in qc.get("players", [])
        if row.get("qc_status") == "accepted" and "embedding" in row
    ]


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def project_tsne(
    embeddings: np.ndarray,
    *,
    perplexity: float,
    random_state: int,
    pca_components: int,
    max_iter: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    features = embeddings
    pca_info: dict[str, Any] = {"enabled": False}
    if pca_components and 2 < pca_components < embeddings.shape[1]:
        pca_components = min(pca_components, embeddings.shape[0] - 1, embeddings.shape[1])
        pca = PCA(n_components=pca_components, random_state=random_state)
        features = pca.fit_transform(embeddings)
        pca_info = {
            "enabled": True,
            "components": int(pca_components),
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        }

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric="cosine",
        init="pca",
        learning_rate="auto",
        max_iter=max_iter,
        random_state=random_state,
    )
    coords = tsne.fit_transform(features)
    return coords.astype(np.float32), pca_info


def project_pca(
    embeddings: np.ndarray,
    *,
    random_state: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(embeddings)
    return coords.astype(np.float32), {
        "enabled": True,
        "components": 2,
        "explained_variance_ratio": [
            float(value) for value in pca.explained_variance_ratio_
        ],
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
    }


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    distances = 1.0 - (embeddings @ embeddings.T)
    distances = np.clip(distances, 0.0, 2.0)
    np.fill_diagonal(distances, 0.0)
    return distances.astype(np.float32)


def project_mds(
    embeddings: np.ndarray,
    *,
    random_state: int,
    max_iter: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.manifold import MDS

    distances = cosine_distance_matrix(embeddings)
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        normalized_stress="auto",
        n_init=1,
        max_iter=max_iter,
        random_state=random_state,
    )
    coords = mds.fit_transform(distances)
    return coords.astype(np.float32), {
        "stress": float(getattr(mds, "stress_", np.nan)),
        "max_iter": max_iter,
    }


def project_isomap(
    embeddings: np.ndarray,
    *,
    n_neighbors: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.manifold import Isomap

    isomap = Isomap(
        n_components=2,
        n_neighbors=n_neighbors,
        metric="cosine",
    )
    coords = isomap.fit_transform(embeddings)
    return coords.astype(np.float32), {
        "n_neighbors": n_neighbors,
        "reconstruction_error": float(isomap.reconstruction_error()),
    }


def project_spectral(
    embeddings: np.ndarray,
    *,
    random_state: int,
    n_neighbors: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.manifold import SpectralEmbedding

    spectral = SpectralEmbedding(
        n_components=2,
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    coords = spectral.fit_transform(embeddings)
    return coords.astype(np.float32), {
        "n_neighbors": n_neighbors,
        "affinity": "nearest_neighbors",
    }


def scale_coords(coords: np.ndarray) -> np.ndarray:
    mins = coords.min(axis=0, keepdims=True)
    maxs = coords.max(axis=0, keepdims=True)
    span = np.where(maxs - mins == 0, 1.0, maxs - mins)
    return (coords - mins) / span


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    qc = json.loads(args.input.read_text(encoding="utf-8"))
    rows = accepted_players(qc)
    if not rows:
        raise ValueError("El QC no contiene jugadores aceptados con embedding.")

    embeddings = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
    normalized = l2_normalize(embeddings)

    params: dict[str, Any]
    if args.method == "tsne":
        coords, pca_info = project_tsne(
            normalized,
            perplexity=args.perplexity,
            random_state=args.random_state,
            pca_components=args.pca_components,
            max_iter=args.max_iter,
        )
        params = {
            "perplexity": args.perplexity,
            "random_state": args.random_state,
            "pca": pca_info,
            "max_iter": args.max_iter,
        }
    elif args.method == "pca":
        coords, pca_info = project_pca(normalized, random_state=args.random_state)
        params = {
            "random_state": args.random_state,
            "pca": pca_info,
        }
    elif args.method == "mds":
        coords, method_info = project_mds(
            normalized,
            random_state=args.random_state,
            max_iter=args.max_iter,
        )
        params = {
            "random_state": args.random_state,
            **method_info,
        }
    elif args.method == "isomap":
        coords, method_info = project_isomap(
            normalized,
            n_neighbors=args.n_neighbors,
        )
        params = method_info
    elif args.method == "spectral":
        coords, method_info = project_spectral(
            normalized,
            random_state=args.random_state,
            n_neighbors=args.n_neighbors,
        )
        params = {
            "random_state": args.random_state,
            **method_info,
        }
    else:
        raise ValueError(f"Metodo no soportado: {args.method}")
    scaled = scale_coords(coords)

    points = []
    for idx, (row, raw, xy) in enumerate(zip(rows, coords, scaled, strict=True)):
        points.append(
            {
                "idx": idx,
                "team": row.get("team"),
                "team_country": row.get("team_country"),
                "number": row.get("number"),
                "position": row.get("position"),
                "name": row.get("name"),
                "local_image": row.get("local_image"),
                "x": float(xy[0]),
                "y": float(xy[1]),
                "raw_x": float(raw[0]),
                "raw_y": float(raw[1]),
            },
        )

    return {
        "schema": "phyloface-vitrina-embedding-projection-v0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_qc_manifest": rel_path(args.input),
        "source_manifest_schema": qc.get("source_manifest_schema"),
        "source_publication_ok": qc.get("source_publication_ok"),
        "source_license_status": qc.get("source_license_status"),
        "embedding_model": qc.get("model_name"),
        "method": args.method,
        "metric": "cosine",
        "players_count": len(points),
        "teams_count": len({point["team"] for point in points}),
        "params": params,
        "points": points,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Proyecta embeddings de jugadores a 2D para la vitrina.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Manifiesto QC.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON de salida.")
    parser.add_argument(
        "--method",
        choices=["tsne", "pca", "mds", "isomap", "spectral"],
        default="tsne",
    )
    parser.add_argument("--perplexity", type=float, default=35.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--n-neighbors", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(args)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Proyeccion escrita: {args.output}")
    print(
        f"{payload['method']} · jugadores: {payload['players_count']} · "
        f"selecciones: {payload['teams_count']}",
    )


if __name__ == "__main__":
    main()
