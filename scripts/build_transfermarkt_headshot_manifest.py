#!/usr/bin/env python3
# =========================================
# ID: PHYLOFACE_TEAMS_TRANSFERMARKT_HEADSHOTS
# VERSION: v0.1
# =========================================
# FILE: scripts/build_transfermarkt_headshot_manifest.py
#
# Toma un manifiesto de jugadores ya generado para la vitrina y busca retratos
# estandarizados en Transfermarkt. Esta fuente se prioriza por consistencia visual
# para comparacion facial, no por aptitud de publicacion. Todo registro queda
# marcado como UNREVIEWED_NONPUBLIC_RESEARCH hasta resolver licencia/permiso.

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urljoin

import requests
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSFERMARKT_BASE = "https://www.transfermarkt.com"
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data/output/teams/manifest_wikimedia_northamerica2026_all_max8_downloaded.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data/output/teams/manifest_transfermarkt_northamerica2026_headshots.json"
)
DEFAULT_IMAGE_DIR = (
    PROJECT_ROOT / "data/input/img/teams_players/northamerica2026_transfermarkt"
)
USER_AGENT = (
    "Mozilla/5.0 (compatible; mendelEmbeddings/0.1; local research prototype)"
)

TEAM_NATIONALITY_ALIASES = {
    "cabo verde": {"cape verde", "cabo verde"},
    "cape verde": {"cape verde", "cabo verde"},
    "congo dr": {"dr congo", "congo dr", "democratic republic of the congo"},
    "cote d'ivoire": {"ivory coast", "cote d'ivoire", "côte d'ivoire"},
    "côte d'ivoire": {"ivory coast", "cote d'ivoire", "côte d'ivoire"},
    "curaçao": {"curacao", "curaçao"},
    "cura-ao": {"curacao", "curaçao"},
    "england": {"england"},
    "iran": {"iran"},
    "korea republic": {"south korea", "korea republic"},
    "south korea": {"south korea", "korea republic"},
    "turkiye": {"turkey", "turkiye", "türkiye"},
    "türkiye": {"turkey", "turkiye", "türkiye"},
    "united states": {"united states", "usa"},
    "usa": {"united states", "usa"},
}


@dataclass
class Candidate:
    name: str
    profile_url: str
    image_url_small: str | None
    club: str | None
    position: str | None
    age: str | None
    nationalities: list[str]
    market_value: str | None
    score: float


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_value = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_value).strip("-").lower()
    return ascii_value or "item"


def normalize(value: str | None) -> str:
    if not value:
        return ""
    text = unicodedata.normalize("NFKD", value)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip().casefold()
    return text


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def clean_player_query(value: Any) -> str:
    text = clean_text(value) or ""
    text = re.sub(r"\s*\(\s*(?:captain|vice-captain)\s*\)\s*", " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def query_variants(player_name: str) -> list[str]:
    variants = [player_name]
    parts = player_name.split()
    east_asian_surnames = {"cho", "jo", "kim", "lee", "park", "seol", "song"}
    if len(parts) >= 2 and normalize(parts[0]) in east_asian_surnames:
        variants.append(" ".join(parts[1:] + parts[:1]))
    return list(dict.fromkeys(item for item in variants if item))


def request_text(session: requests.Session, url: str, retries: int = 3) -> str:
    for attempt in range(retries + 1):
        response = session.get(url, timeout=45)
        if response.status_code not in {429, 500, 502, 503, 504}:
            response.raise_for_status()
            return response.text
        retry_after = response.headers.get("Retry-After")
        delay = float(retry_after) if retry_after and retry_after.isdigit() else min(2.0 * (attempt + 1), 10.0)
        if attempt >= retries:
            response.raise_for_status()
        time.sleep(delay)
    raise RuntimeError("request_text alcanzo un estado imposible.")


def image_variant(url: str | None, size: str) -> str | None:
    if not url or "default.jpg" in url:
        return None
    return re.sub(r"/portrait/(?:small|header|big)/", f"/portrait/{size}/", url)


def team_aliases(team: str) -> set[str]:
    key = normalize(team)
    return TEAM_NATIONALITY_ALIASES.get(key, {key})


def score_candidate(player: dict[str, Any], candidate: Candidate) -> float:
    player_name = clean_player_query(player.get("name"))
    team = clean_text(player.get("team")) or ""
    wanted_name = normalize(player_name)
    candidate_name = normalize(candidate.name)
    wanted_tokens = set(wanted_name.split())
    candidate_tokens = set(candidate_name.split())
    aliases = team_aliases(team)

    score = 0.0
    if candidate_name == wanted_name:
        score += 5.0
    elif wanted_tokens and wanted_tokens == candidate_tokens:
        score += 4.5
    elif wanted_name and (wanted_name in candidate_name or candidate_name in wanted_name):
        score += 2.5

    candidate_nats = {normalize(item) for item in candidate.nationalities}
    if candidate_nats & aliases:
        score += 2.0

    if image_variant(candidate.image_url_small, "big"):
        score += 2.0
    else:
        score -= 2.0

    if candidate.market_value and candidate.market_value != "-":
        score += 0.25
    return score


def parse_candidates(html: str, player: dict[str, Any]) -> list[Candidate]:
    soup = BeautifulSoup(html, "html.parser")
    candidates: list[Candidate] = []
    seen: set[str] = set()

    for link in soup.find_all("a", href=re.compile(r"/profil/spieler/\d+")):
        href = link.get("href") or ""
        profile_url = urljoin(TRANSFERMARKT_BASE, href)
        if profile_url in seen:
            continue
        seen.add(profile_url)

        row = result_row(link)
        if row is None:
            continue
        cells = row.find_all("td", recursive=False)
        inline = link.find_parent("table")
        image = inline.find("img") if inline else row.find("img")
        club_link = inline.find("a", href=re.compile(r"/startseite/verein/")) if inline else None

        name = clean_text(link.get("title")) or clean_text(link.get_text(" ")) or ""
        nationalities = [
            item
            for item in (clean_text(img.get("title")) for img in row.find_all("img", class_=re.compile("flag")))
            if item
        ]
        candidate = Candidate(
            name=name,
            profile_url=profile_url,
            image_url_small=image.get("src") if image else None,
            club=clean_text(club_link.get_text(" ")) if club_link else None,
            position=clean_text(cells[1].get_text(" ")) if len(cells) > 1 else None,
            age=clean_text(cells[3].get_text(" ")) if len(cells) > 3 else None,
            nationalities=nationalities,
            market_value=clean_text(cells[5].get_text(" ")) if len(cells) > 5 else None,
            score=0.0,
        )
        candidate.score = score_candidate(player, candidate)
        candidates.append(candidate)

    return sorted(candidates, key=lambda item: item.score, reverse=True)


def result_row(link: Any) -> Any | None:
    """Devuelve la fila externa de resultados, no la fila interna del mini-table."""
    node = link
    while node is not None:
        node = node.parent
        if getattr(node, "name", None) != "tr":
            continue
        cells = node.find_all("td", recursive=False)
        if len(cells) >= 5:
            return node
    return None


def search_player(session: requests.Session, player: dict[str, Any]) -> dict[str, Any]:
    query = clean_player_query(player.get("name"))
    team = clean_text(player.get("team")) or ""
    candidates_by_url = {}
    query_urls = []
    for variant in query_variants(query):
        url = f"{TRANSFERMARKT_BASE}/schnellsuche/ergebnis/schnellsuche?query={quote_plus(variant)}"
        query_urls.append(url)
        html = request_text(session, url)
        for candidate in parse_candidates(html, player):
            current = candidates_by_url.get(candidate.profile_url)
            if current is None or candidate.score > current.score:
                candidates_by_url[candidate.profile_url] = candidate
        time.sleep(0.05)
    candidates = sorted(candidates_by_url.values(), key=lambda item: item.score, reverse=True)
    best = next((item for item in candidates if image_variant(item.image_url_small, "big")), None)

    if not best or best.score < 5.0:
        return {
            "status": "missing_confident_match",
            "query_url": query_urls[0] if query_urls else None,
            "query_urls": query_urls,
            "best_score": best.score if best else None,
            "candidates": [candidate_payload(item) for item in candidates[:5]],
            "selected": None,
        }

    selected = candidate_payload(best)
    selected["image_url_header"] = image_variant(best.image_url_small, "header")
    selected["image_url_big"] = image_variant(best.image_url_small, "big")
    return {
        "status": "image_found",
        "query_url": query_urls[0] if query_urls else None,
        "query_urls": query_urls,
        "best_score": best.score,
        "candidates": [candidate_payload(item) for item in candidates[:5]],
        "selected": selected,
        "source_policy": {
            "license_status": "UNREVIEWED_NONPUBLIC_RESEARCH",
            "publication_ok": False,
            "reason": (
                "Transfermarkt se usa aqui por estandarizacion visual. "
                "No publicar ni redistribuir sin resolver permisos/licencia."
            ),
        },
    }


def candidate_payload(candidate: Candidate) -> dict[str, Any]:
    return {
        "name": candidate.name,
        "profile_url": candidate.profile_url,
        "image_url_small": candidate.image_url_small,
        "club": candidate.club,
        "position": candidate.position,
        "age": candidate.age,
        "nationalities": candidate.nationalities,
        "market_value": candidate.market_value,
        "match_score": candidate.score,
    }


def iter_players(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    players: list[dict[str, Any]] = []
    for team in manifest.get("teams", []):
        team_name = team.get("team")
        for player in team.get("players", []):
            item = dict(player)
            item["team"] = item.get("team") or team_name
            players.append(item)
    return players


def download_image(session: requests.Session, url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return
    response = session.get(url, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    source = json.loads(args.input.read_text(encoding="utf-8"))
    players = iter_players(source)
    if args.max_players is not None:
        players = players[: args.max_players]

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    rows = []
    for idx, player in enumerate(players, start=1):
        name = clean_text(player.get("name")) or "?"
        team = clean_text(player.get("team")) or "?"
        print(f"[{idx}/{len(players)}] {team} - {name}", flush=True)
        resolution = search_player(session, player)
        local_image = None

        selected = resolution.get("selected") or {}
        image_url = selected.get("image_url_big") or selected.get("image_url_header")
        if args.download_images and image_url:
            suffix = Path(image_url.split("?", 1)[0]).suffix or ".jpg"
            local_path = args.image_dir / slugify(team) / f"{slugify(name)}{suffix.lower()}"
            download_image(session, image_url, local_path)
            local_image = str(local_path.relative_to(PROJECT_ROOT))

        rows.append(
            {
                "team": team,
                "number": player.get("number"),
                "position": player.get("position"),
                "name": name,
                "source_player": {
                    "club": player.get("club"),
                    "wikidata": (player.get("resolution") or {}).get("wikidata"),
                },
                "resolution": resolution,
                "local_image": local_image,
            }
        )
        time.sleep(args.sleep)

    return {
        "schema": "phyloface-transfermarkt-headshot-manifest-v0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.input),
        "source_priority": "standardized_headshot_over_license",
        "sources": {
            "transfermarkt_search": f"{TRANSFERMARKT_BASE}/schnellsuche/ergebnis/schnellsuche",
            "transfermarkt_profile_image_criteria": (
                "https://www.transfermarkt.co.uk/agent-support/addProfilePictures/berater"
            ),
        },
        "download_images": bool(args.download_images),
        "players_count": len(rows),
        "image_found_count": sum(1 for row in rows if row["resolution"]["status"] == "image_found"),
        "players": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Busca retratos estandarizados Transfermarkt para jugadores de la vitrina.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Manifiesto base con teams[].players[].")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON de salida.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR, help="Destino para --download-images.")
    parser.add_argument("--download-images", action="store_true", help="Descarga los retratos seleccionados.")
    parser.add_argument("--max-players", type=int, default=None, help="Limite total para smoke/piloto.")
    parser.add_argument("--sleep", type=float, default=0.25, help="Pausa entre busquedas.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    args.image_dir = args.image_dir.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(args)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manifest escrito: {args.output}")
    print(f"Imagenes: {manifest['image_found_count']}/{manifest['players_count']}")


if __name__ == "__main__":
    main()
