#!/usr/bin/env python3
# =========================================
# ID: PHYLOFACE_VITRINA_COVERAGE_REPORT
# VERSION: v0.1
# =========================================
# FILE: scripts/report_vitrina_coverage.py
#
# Reporte local de cobertura para priorizar arreglos de retratos/QC en la
# vitrina. No descarga nada: resume manifiesto Transfermarkt + QC.

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HEADSHOTS = (
    PROJECT_ROOT
    / "data/output/teams/manifest_transfermarkt_northamerica2026_headshots.json"
)
DEFAULT_QC = (
    PROJECT_ROOT
    / "data/output/teams/manifest_transfermarkt_northamerica2026_headshots_qc.json"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "data/output/teams/coverage_report"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def team_key(row: dict[str, Any]) -> str:
    return str(row.get("team") or "UNKNOWN")


def player_key(row: dict[str, Any]) -> tuple[str, str]:
    return (team_key(row), str(row.get("name") or "UNKNOWN"))


def summarize(headshots: dict[str, Any], qc: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    headshot_rows = headshots.get("players", [])
    qc_rows = qc.get("players", [])
    qc_by_player = {player_key(row): row for row in qc_rows}
    teams: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "team": None,
        "source_players": 0,
        "image_found": 0,
        "local_image": 0,
        "qc_accepted": 0,
        "qc_rejected": 0,
        "reject_reasons": Counter(),
        "rejected_players": [],
    })

    for row in headshot_rows:
        team = team_key(row)
        item = teams[team]
        item["team"] = team
        item["source_players"] += 1
        if (row.get("resolution") or {}).get("status") == "image_found":
            item["image_found"] += 1
        if row.get("local_image"):
            item["local_image"] += 1

        qc_row = qc_by_player.get(player_key(row))
        if not qc_row:
            continue
        if qc_row.get("qc_status") == "accepted":
            item["qc_accepted"] += 1
        else:
            item["qc_rejected"] += 1
            reasons = qc_row.get("reject_reasons") or ["unknown_reject"]
            for reason in reasons:
                item["reject_reasons"][reason] += 1
            item["rejected_players"].append({
                "team": team,
                "name": row.get("name"),
                "position": row.get("position"),
                "local_image": row.get("local_image"),
                "reject_reasons": ";".join(reasons),
                "error": qc_row.get("error"),
                "best_score": (row.get("resolution") or {}).get("best_score"),
                "query_url": (row.get("resolution") or {}).get("query_url"),
            })

    summaries = []
    rejected = []
    for item in teams.values():
        source_players = item["source_players"]
        accepted = item["qc_accepted"]
        found = item["image_found"]
        accepted_rate = accepted / source_players if source_players else 0.0
        found_rate = found / source_players if source_players else 0.0
        reason_counts = dict(sorted(item["reject_reasons"].items()))
        priority_score = (
            (source_players - accepted) * 10
            + item["reject_reasons"].get("missing_local_image", 0) * 6
            + item["reject_reasons"].get("image_read_error", 0) * 5
            + item["reject_reasons"].get("no_face_detected", 0) * 3
            + item["reject_reasons"].get("not_exactly_one_face", 0) * 2
        )
        summaries.append({
            "team": item["team"],
            "source_players": source_players,
            "image_found": found,
            "local_image": item["local_image"],
            "qc_accepted": accepted,
            "qc_rejected": item["qc_rejected"],
            "accepted_rate": accepted_rate,
            "image_found_rate": found_rate,
            "priority_score": priority_score,
            "reject_reasons": reason_counts,
            "missing_local_image": reason_counts.get("missing_local_image", 0),
            "no_face_detected": reason_counts.get("no_face_detected", 0),
            "image_read_error": reason_counts.get("image_read_error", 0),
            "not_exactly_one_face": reason_counts.get("not_exactly_one_face", 0),
        })
        rejected.extend(item["rejected_players"])

    summaries.sort(key=lambda row: (-row["priority_score"], row["qc_accepted"], row["team"]))
    rejected.sort(key=lambda row: (row["team"], row["name"] or ""))
    return summaries, rejected


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def html_table(rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    head = "".join(f"<th>{html.escape(name)}</th>" for name in fieldnames)
    body = []
    for row in rows:
        cells = []
        for name in fieldnames:
            value = row.get(name)
            if isinstance(value, float):
                value = f"{value:.1%}" if name.endswith("_rate") else f"{value:.4f}"
            elif isinstance(value, dict):
                value = ", ".join(f"{k}={v}" for k, v in value.items())
            cells.append(f"<td>{html.escape(str(value if value is not None else ''))}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(path: Path, summaries: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> None:
    summary_fields = [
        "team",
        "source_players",
        "image_found",
        "qc_accepted",
        "qc_rejected",
        "accepted_rate",
        "priority_score",
        "missing_local_image",
        "no_face_detected",
        "image_read_error",
        "not_exactly_one_face",
    ]
    rejected_fields = [
        "team",
        "name",
        "position",
        "reject_reasons",
        "local_image",
        "error",
    ]
    top = summaries[:12]
    path.write_text(
        f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Vitrina 2026 - cobertura QC</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #17202a; }}
    h1, h2 {{ margin-bottom: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 13px; }}
    th, td {{ border: 1px solid #d8dde6; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f6f8; position: sticky; top: 0; }}
    td:nth-child(n+2) {{ white-space: nowrap; }}
    .muted {{ color: #667085; }}
  </style>
</head>
<body>
  <h1>Vitrina 2026 - cobertura QC</h1>
  <p class="muted">Generado {datetime.now(timezone.utc).isoformat()} UTC. Prioridad = faltantes/rechazos ponderados para arreglar cobertura.</p>
  <h2>Prioridad por selección</h2>
  {html_table(top, summary_fields)}
  <h2>Todas las selecciones</h2>
  {html_table(summaries, summary_fields)}
  <h2>Jugadores rechazados</h2>
  {html_table(rejected, rejected_fields)}
</body>
</html>
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reporte de cobertura QC de vitrina.")
    parser.add_argument("--headshots", type=Path, default=DEFAULT_HEADSHOTS)
    parser.add_argument("--qc", type=Path, default=DEFAULT_QC)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries, rejected = summarize(load_json(args.headshots), load_json(args.qc))

    write_csv(
        args.out_dir / "coverage_by_team.csv",
        summaries,
        [
            "team",
            "source_players",
            "image_found",
            "local_image",
            "qc_accepted",
            "qc_rejected",
            "accepted_rate",
            "image_found_rate",
            "priority_score",
            "missing_local_image",
            "no_face_detected",
            "image_read_error",
            "not_exactly_one_face",
            "reject_reasons",
        ],
    )
    write_csv(
        args.out_dir / "rejected_players.csv",
        rejected,
        [
            "team",
            "name",
            "position",
            "local_image",
            "reject_reasons",
            "error",
            "best_score",
            "query_url",
        ],
    )
    write_html(args.out_dir / "coverage_report.html", summaries, rejected)
    print(f"Reporte escrito en: {args.out_dir}")
    print("Top prioridad:")
    for row in summaries[:10]:
        print(
            f"- {row['team']}: accepted {row['qc_accepted']}/{row['source_players']} "
            f"priority={row['priority_score']} reasons={row['reject_reasons']}"
        )


if __name__ == "__main__":
    main()
