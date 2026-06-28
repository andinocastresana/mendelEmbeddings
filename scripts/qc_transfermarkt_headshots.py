#!/usr/bin/env python3
# =========================================
# ID: PHYLOFACE_TEAMS_TRANSFERMARKT_HEADSHOT_QC
# VERSION: v0.1
# =========================================
# FILE: scripts/qc_transfermarkt_headshots.py
#
# QC facial offline para los retratos estandarizados de la vitrina Mundial 2026.
# Toma el manifiesto Transfermarkt y separa fotos aceptadas/rechazadas antes de
# generar embeddings y matrices publicables/client-side.

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data/output/teams/manifest_transfermarkt_northamerica2026_headshots.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data/output/teams/manifest_transfermarkt_northamerica2026_headshots_qc.json"
)


def load_face_detector(args: argparse.Namespace):
    from phyloface.core.detector import FaceDetector

    return FaceDetector(
        library_name="insightface",
        model_name=args.model_name,
        det_size=(args.det_size, args.det_size),
        det_thresh=args.det_thresh,
        ctx_id=args.ctx_id,
        max_faces=args.max_faces,
    )


def iter_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("schema") == "phyloface-fifa-official-headshot-manifest-v0.1":
        return fifa_rows(manifest)
    return list(manifest.get("players", []))


def slug(text: Any) -> str:
    if not text:
        return "unknown"
    normalized = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    value = re.sub(r"[^a-zA-Z0-9]+", "-", normalized).strip("-").lower()
    return value or "unknown"


def fifa_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    image_root = Path("data/input/img/teams_players/northamerica2026_fifa_official")
    source_policy = {
        "license_status": manifest.get("license_status"),
        "publication_ok": manifest.get("publication_ok"),
        "license_note": manifest.get("license_note"),
        "source": manifest.get("source"),
    }
    for team in manifest.get("teams", []):
        team_name = team.get("team_name")
        team_slug = slug(team_name)
        for player in team.get("players", []):
            local_image = None
            if player.get("has_photo") and player.get("id_player"):
                filename = f"{slug(player.get('name'))}_{player.get('id_player')}.png"
                local_image = str(image_root / team_slug / filename)
            rows.append(
                {
                    "team": team_name,
                    "team_country": team.get("country"),
                    "team_id": team.get("id_team"),
                    "number": player.get("jersey_number"),
                    "position": player.get("position"),
                    "real_position": player.get("real_position"),
                    "position_code": player.get("position_code"),
                    "name": player.get("name"),
                    "short_name": player.get("short_name"),
                    "id_player": player.get("id_player"),
                    "country": player.get("country"),
                    "birth_date": player.get("birth_date"),
                    "height_cm": player.get("height_cm"),
                    "weight_kg": player.get("weight_kg"),
                    "local_image": local_image,
                    "photo_url_best": player.get("photo_url_best"),
                    "source_policy": source_policy,
                },
            )
    return rows


def rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def image_quality(path: Path) -> dict[str, Any]:
    import cv2
    from phyloface.core.detector import FaceDetector

    image = FaceDetector.read_image_bgr(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    height, width = image.shape[:2]
    return {
        "image_width": int(width),
        "image_height": int(height),
        "sharpness_laplacian_var": laplacian_var,
    }


def face_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    bboxes = np.asarray(payload["bboxes"])
    det_scores = np.asarray(payload["det_scores"])
    kps = np.asarray(payload["kps"])
    image_width = int(payload["image_width"])
    image_height = int(payload["image_height"])

    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    selected_idx = int(np.argmax(areas))
    x1, y1, x2, y2 = [int(value) for value in bboxes[selected_idx]]
    bbox_width = max(0, x2 - x1)
    bbox_height = max(0, y2 - y1)
    image_area = max(1, image_width * image_height)
    bbox_area_ratio = float((bbox_width * bbox_height) / image_area)
    bbox_width_ratio = float(bbox_width / max(1, image_width))
    bbox_height_ratio = float(bbox_height / max(1, image_height))
    sorted_areas = np.sort(areas.astype(np.float32))[::-1]
    secondary_area_ratio = (
        float(sorted_areas[1] / image_area)
        if sorted_areas.size > 1 else 0.0
    )
    secondary_to_primary_area_ratio = (
        float(sorted_areas[1] / sorted_areas[0])
        if sorted_areas.size > 1 and sorted_areas[0] > 0 else 0.0
    )

    clipped_margin_px = 2
    clipped = (
        x1 <= clipped_margin_px
        or y1 <= clipped_margin_px
        or x2 >= image_width - clipped_margin_px
        or y2 >= image_height - clipped_margin_px
    )

    roll_degrees = None
    selected_kps = kps[selected_idx]
    if selected_kps.shape[0] >= 2 and np.isfinite(selected_kps[:2]).all():
        left_eye, right_eye = selected_kps[0], selected_kps[1]
        dy = float(right_eye[1] - left_eye[1])
        dx = float(right_eye[0] - left_eye[0])
        roll_degrees = abs(math.degrees(math.atan2(dy, dx)))

    return {
        "n_faces": int(payload["n_faces"]),
        "selected_face_index": selected_idx,
        "bbox": [x1, y1, x2, y2],
        "bbox_width_ratio": bbox_width_ratio,
        "bbox_height_ratio": bbox_height_ratio,
        "bbox_area_ratio": bbox_area_ratio,
        "secondary_area_ratio": secondary_area_ratio,
        "secondary_to_primary_area_ratio": secondary_to_primary_area_ratio,
        "bbox_clipped": bool(clipped),
        "det_score": float(det_scores[selected_idx]),
        "roll_degrees": roll_degrees,
    }


def rejection_reasons(metrics: dict[str, Any], args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    dominant_single_face = (
        metrics.get("n_faces") == 1
        or (
            metrics.get("n_faces", 0) > 1
            and metrics.get("secondary_to_primary_area_ratio", 1.0)
            <= args.max_secondary_to_primary_area_ratio
            and metrics.get("secondary_area_ratio", 1.0)
            <= args.max_secondary_area_ratio
        )
    )
    if not dominant_single_face:
        reasons.append("not_exactly_one_face")
    if metrics.get("det_score", 0.0) < args.min_det_score:
        reasons.append("low_detection_score")
    if metrics.get("bbox_area_ratio", 0.0) < args.min_bbox_area_ratio:
        reasons.append("face_too_small")
    if metrics.get("bbox_area_ratio", 0.0) > args.max_bbox_area_ratio:
        reasons.append("face_too_large")
    if metrics.get("bbox_clipped"):
        reasons.append("face_bbox_clipped")
    roll_degrees = metrics.get("roll_degrees")
    if roll_degrees is not None and roll_degrees > args.max_roll_degrees:
        reasons.append("excessive_roll")
    if metrics.get("sharpness_laplacian_var", 0.0) < args.min_sharpness:
        reasons.append("low_sharpness")
    return reasons


def embedding_payload(payload: dict[str, Any], selected_face_index: int) -> list[float]:
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    return embeddings[selected_face_index].astype(float).tolist()


def qc_one(row: dict[str, Any], detector: Any, args: argparse.Namespace) -> dict[str, Any]:
    local_image = row.get("local_image")
    base = {
        "team": row.get("team"),
        "team_country": row.get("team_country"),
        "team_id": row.get("team_id"),
        "number": row.get("number"),
        "position": row.get("position"),
        "real_position": row.get("real_position"),
        "position_code": row.get("position_code"),
        "name": row.get("name"),
        "short_name": row.get("short_name"),
        "id_player": row.get("id_player"),
        "country": row.get("country"),
        "birth_date": row.get("birth_date"),
        "height_cm": row.get("height_cm"),
        "weight_kg": row.get("weight_kg"),
        "local_image": local_image,
        "source_policy": row.get("source_policy")
        or (row.get("resolution") or {}).get("source_policy"),
    }
    if not local_image:
        return {
            **base,
            "qc_status": "rejected",
            "reject_reasons": ["missing_local_image"],
            "metrics": {},
        }

    image_path = PROJECT_ROOT / local_image
    if not image_path.exists():
        return {
            **base,
            "qc_status": "rejected",
            "reject_reasons": ["local_image_not_found"],
            "metrics": {},
        }

    try:
        quality = image_quality(image_path)
        payload = detector.extract_faces_payload(image_path)
        metrics = {**quality, **face_metrics(payload)}
        reasons = rejection_reasons(metrics, args)
    except Exception as exc:  # noqa: BLE001 - QC debe reportar y seguir el lote.
        reason = processing_error_reason(exc)
        return {
            **base,
            "qc_status": "rejected",
            "reject_reasons": [reason],
            "error": str(exc),
            "metrics": {},
        }

    result = {
        **base,
        "qc_status": "accepted" if not reasons else "rejected",
        "reject_reasons": reasons,
        "metrics": metrics,
    }
    if args.include_embeddings and not reasons:
        result["embedding"] = embedding_payload(payload, metrics["selected_face_index"])
        result["embedding_model"] = args.model_name
    return result


def processing_error_reason(exc: Exception) -> str:
    message = str(exc)
    if message.startswith("No se detectaron caras"):
        return "no_face_detected"
    if message.startswith("No se pudo leer la imagen"):
        return "image_read_error"
    return "processing_error"


def build_qc_manifest(args: argparse.Namespace) -> dict[str, Any]:
    source = json.loads(args.input.read_text(encoding="utf-8"))
    rows = iter_rows(source)
    if args.max_players is not None:
        rows = rows[: args.max_players]

    detector = load_face_detector(args)
    results = []
    for idx, row in enumerate(rows, start=1):
        print(f"[{idx}/{len(rows)}] {row.get('team')} - {row.get('name')}", flush=True)
        results.append(qc_one(row, detector, args))

    reason_counts = Counter(
        reason
        for row in results
        for reason in row.get("reject_reasons", [])
    )
    accepted_count = sum(1 for row in results if row["qc_status"] == "accepted")

    return {
        "schema": "phyloface-transfermarkt-headshot-qc-v0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": rel_path(args.input),
        "source_manifest_schema": source.get("schema"),
        "source_publication_ok": source.get("publication_ok"),
        "source_license_status": source.get("license_status"),
        "model_name": args.model_name,
        "det_size": [args.det_size, args.det_size],
        "det_thresh": args.det_thresh,
        "ctx_id": args.ctx_id,
        "thresholds": {
            "min_det_score": args.min_det_score,
            "min_bbox_area_ratio": args.min_bbox_area_ratio,
            "max_bbox_area_ratio": args.max_bbox_area_ratio,
            "max_roll_degrees": args.max_roll_degrees,
            "min_sharpness": args.min_sharpness,
            "max_secondary_to_primary_area_ratio": args.max_secondary_to_primary_area_ratio,
            "max_secondary_area_ratio": args.max_secondary_area_ratio,
        },
        "include_embeddings": bool(args.include_embeddings),
        "players_count": len(results),
        "accepted_count": accepted_count,
        "rejected_count": len(results) - accepted_count,
        "reject_reason_counts": dict(sorted(reason_counts.items())),
        "players": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QC facial para retratos Transfermarkt de la vitrina.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Manifiesto Transfermarkt.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON de salida QC.")
    parser.add_argument("--model-name", default="buffalo_l", help="Modelo InsightFace.")
    parser.add_argument("--det-size", type=int, default=640, help="Lado del detector InsightFace.")
    parser.add_argument("--det-thresh", type=float, default=0.20, help="Umbral de deteccion InsightFace.")
    parser.add_argument("--ctx-id", type=int, default=-1, help="-1 CPU, >=0 GPU.")
    parser.add_argument("--max-faces", type=int, default=8, help="Maximo de caras a reportar por imagen.")
    parser.add_argument("--max-players", type=int, default=None, help="Limite para smoke/piloto.")
    parser.add_argument("--include-embeddings", action="store_true", help="Incluye embedding full-face para aceptados.")
    parser.add_argument("--min-det-score", type=float, default=0.20)
    parser.add_argument("--min-bbox-area-ratio", type=float, default=0.08)
    parser.add_argument("--max-bbox-area-ratio", type=float, default=0.90)
    parser.add_argument("--max-roll-degrees", type=float, default=18.0)
    parser.add_argument("--min-sharpness", type=float, default=20.0)
    parser.add_argument("--max-secondary-to-primary-area-ratio", type=float, default=0.08)
    parser.add_argument("--max-secondary-area-ratio", type=float, default=0.04)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    manifest = build_qc_manifest(args)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"QC escrito: {args.output}")
    print(f"Aceptadas: {manifest['accepted_count']}/{manifest['players_count']}")
    print(f"Rechazos: {manifest['reject_reason_counts']}")


if __name__ == "__main__":
    main()
