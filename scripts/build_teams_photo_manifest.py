#!/usr/bin/env python3
# =========================================
# ID: PHYLOFACE_TEAMS_MANIFEST_001
# VERSION: v0.1
# =========================================
# FILE: scripts/build_teams_photo_manifest.py
#
# Genera un manifiesto auditable de jugadores + imagenes Wikimedia/Commons para
# la vitrina de selecciones. Por defecto NO descarga imagenes: primero produce
# metadata para revisar cobertura, licencias y calidad.

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import requests
from requests import HTTPError
from bs4 import BeautifulSoup

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - mensaje de entorno
    raise SystemExit(
        "Este script requiere pandas. Activar el entorno del proyecto o instalar "
        "las dependencias de requirements.txt."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data/output/teams/manifest_wikimedia_qatar2022.json"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data/input/img/teams_players/qatar2022"

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
TOURNAMENTS = {
    "northamerica2026": {
        "label": "FIFA World Cup 2026",
        "wikipedia_squads_page": "2026 FIFA World Cup squads",
        "wikipedia_squads_url": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads",
        "fifa_squads_url": "https://www.fifa.com/en/articles/all-world-cup-squad-announcements",
        "default_output": PROJECT_ROOT
        / "data/output/teams/manifest_wikimedia_northamerica2026.json",
        "default_image_dir": PROJECT_ROOT / "data/input/img/teams_players/northamerica2026",
        "squad_status": (
            "provisional_until_2026-06-02; FIFA final squad publication expected "
            "after association submissions due 2026-06-01"
        ),
    },
    "qatar2022": {
        "label": "FIFA World Cup Qatar 2022",
        "wikipedia_squads_page": "2022 FIFA World Cup squads",
        "wikipedia_squads_url": "https://en.wikipedia.org/wiki/2022_FIFA_World_Cup_squads",
        "fifa_squads_url": "https://fdp.fifa.org/assetspublic/ce44/pdf/SquadLists-English.pdf",
        "default_output": DEFAULT_OUTPUT,
        "default_image_dir": DEFAULT_IMAGE_DIR,
        "squad_status": "final_historical",
    },
}

USER_AGENT = (
    "mendelEmbeddings/0.1 "
    "(local research prototype; contact: local-user; Wikimedia-friendly scraper)"
)


@dataclass
class SquadPlayer:
    team: str
    number: str | None
    position: str | None
    name: str
    date_of_birth: str | None
    caps: str | None
    goals: str | None
    club: str | None
    enwiki_title: str | None = None


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_value = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_value).strip("-").lower()
    return ascii_value or "item"


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if text.lower() == "nan":
        return None
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def clean_player_name(value: Any) -> str | None:
    text = clean_text(value)
    if not text:
        return None
    text = re.sub(r"\s*\((?:captain|vice-captain)\)\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def normalize_col(col: Any) -> str:
    if isinstance(col, tuple):
        col = " ".join(str(part) for part in col if str(part) != "nan")
    text = clean_text(col) or ""
    return text.lower().replace(".", "").replace(" ", "_")


def request_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    retries: int = 4,
) -> dict[str, Any]:
    for attempt in range(retries + 1):
        response = session.get(url, params=params, timeout=30)
        if response.status_code not in {429, 500, 502, 503, 504}:
            response.raise_for_status()
            return response.json()

        retry_after = response.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            delay = float(retry_after)
        else:
            delay = min(2.0 * (attempt + 1), 10.0)

        if attempt >= retries:
            response.raise_for_status()
        time.sleep(delay)

    raise RuntimeError("request_json alcanzo un estado imposible.")


def fetch_squads_page_html(session: requests.Session, squads_url: str) -> str:
    response = session.get(squads_url, timeout=60)
    response.raise_for_status()
    return response.text


def normalize_team_name(team: str) -> str:
    wanted = team.casefold()
    aliases = {
        "usa": "united states",
        "us": "united states",
        "england": "england",
        "spain": "spain",
        "france": "france",
        "argentina": "argentina",
    }
    return aliases.get(wanted, wanted)


def find_team_table_html(page_html: str, team: str) -> str | None:
    wanted = normalize_team_name(team)
    soup = BeautifulSoup(page_html, "html.parser")

    for heading in soup.find_all(["h2", "h3"]):
        heading_text = clean_text(heading.get_text(" ")) or ""
        # Remove edit-link noise that appears in some parser outputs.
        heading_text = re.sub(r"\s*\[\s*edit\s*\]\s*", "", heading_text, flags=re.I)
        if heading_text.casefold() != wanted:
            continue

        for sibling in heading.find_all_next():
            if sibling is heading:
                continue
            if sibling.name in {"h2", "h3"}:
                return None
            if sibling.name == "table":
                return str(sibling)
    return None


def extract_squad_table(html: str) -> pd.DataFrame:
    tables = pd.read_html(StringIO(html))
    for table in tables:
        cols = [normalize_col(col) for col in table.columns]
        if any(col in {"player", "player_name"} for col in cols):
            table = table.copy()
            table.columns = cols
            return table
    raise ValueError("No se encontro una tabla de jugadores en la seccion.")


def extract_player_wiki_titles(html: str) -> list[str | None]:
    soup = BeautifulSoup(html, "html.parser")
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue

        header_cells = rows[0].find_all(["th", "td"])
        headers = [clean_text(cell.get_text(" ")) or "" for cell in header_cells]
        normalized = [normalize_col(header) for header in headers]
        if "player" not in normalized:
            continue
        player_idx = normalized.index("player")

        titles: list[str | None] = []
        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if len(cells) <= player_idx:
                continue
            link = cells[player_idx].find("a", href=re.compile(r"^/wiki/"))
            if not link:
                titles.append(None)
                continue
            href = link.get("href", "")
            title = unquote(href.removeprefix("/wiki/")).replace("_", " ")
            if ":" in title:
                titles.append(None)
            else:
                titles.append(title)
        return titles
    return []


def table_to_players(team: str, table: pd.DataFrame) -> list[SquadPlayer]:
    def get(row: Any, *names: str) -> str | None:
        for name in names:
            if name in table.columns:
                return clean_text(row.get(name))
        return None

    players: list[SquadPlayer] = []
    for _, row in table.iterrows():
        name = clean_player_name(get(row, "player", "player_name"))
        if not name or name.lower() in {"player", "head coach"}:
            continue
        players.append(
            SquadPlayer(
                team=team,
                number=get(row, "no", "number"),
                position=get(row, "pos", "position"),
                name=name,
                date_of_birth=get(row, "date_of_birth_(age)", "date_of_birth"),
                caps=get(row, "caps"),
                goals=get(row, "goals"),
                club=get(row, "club"),
            )
        )
    return players


def attach_wiki_titles(players: list[SquadPlayer], titles: list[str | None]) -> None:
    for player, title in zip(players, titles):
        player.enwiki_title = title


def search_wikidata_player(
    session: requests.Session,
    player_name: str,
    team: str,
) -> dict[str, Any] | None:
    queries = [f"{player_name} {team} football", player_name]
    for query in queries:
        payload = request_json(
            session,
            WIKIDATA_API,
            {
                "action": "wbsearchentities",
                "search": query,
                "language": "en",
                "format": "json",
                "limit": 5,
            },
        )
        candidates = payload.get("search", [])
        if not candidates:
            continue

        footballish = [
            item
            for item in candidates
            if "football" in (item.get("description") or "").casefold()
            or "soccer" in (item.get("description") or "").casefold()
        ]
        return (footballish or candidates)[0]
    return None


def get_wikidata_entity(session: requests.Session, qid: str) -> dict[str, Any]:
    payload = request_json(
        session,
        WIKIDATA_API,
        {
            "action": "wbgetentities",
            "ids": qid,
            "props": "claims|labels|sitelinks|descriptions",
            "languages": "en|es",
            "format": "json",
        },
    )
    return payload["entities"][qid]


def enwiki_titles_to_qids(
    session: requests.Session,
    titles: list[str],
) -> dict[str, str]:
    if not titles:
        return {}

    result: dict[str, str] = {}
    chunk_size = 50
    for i in range(0, len(titles), chunk_size):
        chunk = titles[i : i + chunk_size]
        payload = request_json(
            session,
            WIKIPEDIA_API,
            {
                "action": "query",
                "prop": "pageprops",
                "titles": "|".join(chunk),
                "format": "json",
                "redirects": 1,
            },
        )
        normalized_map = {
            item.get("from"): item.get("to")
            for item in payload.get("query", {}).get("normalized", [])
        }
        redirects_map = {
            item.get("from"): item.get("to")
            for item in payload.get("query", {}).get("redirects", [])
        }
        pages = payload.get("query", {}).get("pages", {})
        title_to_qid = {
            page.get("title"): page.get("pageprops", {}).get("wikibase_item")
            for page in pages.values()
        }
        for original in chunk:
            resolved = redirects_map.get(normalized_map.get(original, original), normalized_map.get(original, original))
            qid = title_to_qid.get(resolved)
            if qid:
                result[original] = qid
        time.sleep(0.2)
    return result


def get_wikidata_entities(
    session: requests.Session,
    qids: list[str],
) -> dict[str, dict[str, Any]]:
    if not qids:
        return {}

    entities: dict[str, dict[str, Any]] = {}
    chunk_size = 50
    for i in range(0, len(qids), chunk_size):
        chunk = qids[i : i + chunk_size]
        payload = request_json(
            session,
            WIKIDATA_API,
            {
                "action": "wbgetentities",
                "ids": "|".join(chunk),
                "props": "claims|labels|sitelinks|descriptions",
                "languages": "en|es",
                "format": "json",
            },
        )
        entities.update(payload.get("entities", {}))
        time.sleep(0.2)
    return entities


def claim_value(entity: dict[str, Any], prop: str) -> Any | None:
    claims = entity.get("claims", {}).get(prop, [])
    if not claims:
        return None
    mainsnak = claims[0].get("mainsnak", {})
    datavalue = mainsnak.get("datavalue")
    if not datavalue:
        return None
    return datavalue.get("value")


def commons_imageinfo(session: requests.Session, filename: str) -> dict[str, Any] | None:
    title = f"File:{filename}"
    payload = request_json(
        session,
        COMMONS_API,
        {
            "action": "query",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|mime|size",
            "iiurlwidth": 768,
            "format": "json",
        },
    )
    pages = payload.get("query", {}).get("pages", {})
    for page in pages.values():
        infos = page.get("imageinfo", [])
        if infos:
            return infos[0]
    return None


def commons_imageinfos(
    session: requests.Session,
    filenames: list[str],
) -> dict[str, dict[str, Any]]:
    if not filenames:
        return {}

    result: dict[str, dict[str, Any]] = {}
    chunk_size = 50
    for i in range(0, len(filenames), chunk_size):
        chunk = filenames[i : i + chunk_size]
        titles = [f"File:{filename}" for filename in chunk]
        payload = request_json(
            session,
            COMMONS_API,
            {
                "action": "query",
                "titles": "|".join(titles),
                "prop": "imageinfo",
                "iiprop": "url|extmetadata|mime|size",
                "iiurlwidth": 768,
                "format": "json",
            },
        )
        pages = payload.get("query", {}).get("pages", {})
        for page in pages.values():
            title = page.get("title", "")
            filename = title.removeprefix("File:")
            infos = page.get("imageinfo", [])
            if filename and infos:
                result[filename] = infos[0]
        time.sleep(0.2)
    return result


def extmetadata_value(info: dict[str, Any], key: str) -> str | None:
    raw = info.get("extmetadata", {}).get(key, {}).get("value")
    return clean_text(raw)


def resolve_player_image(
    session: requests.Session,
    player: SquadPlayer,
    sleep_seconds: float,
) -> dict[str, Any]:
    search_hit = search_wikidata_player(session, player.name, player.team)
    if not search_hit:
        return {
            "status": "missing_wikidata",
            "wikidata": None,
            "commons": None,
        }

    qid = search_hit["id"]
    time.sleep(sleep_seconds)
    entity = get_wikidata_entity(session, qid)
    image_name = claim_value(entity, "P18")
    enwiki = entity.get("sitelinks", {}).get("enwiki", {}).get("title")

    wikidata = {
        "id": qid,
        "label": entity.get("labels", {}).get("en", {}).get("value"),
        "description": entity.get("descriptions", {}).get("en", {}).get("value"),
        "concepturi": search_hit.get("concepturi"),
        "enwiki_title": enwiki,
        "enwiki_url": f"https://en.wikipedia.org/wiki/{quote(enwiki.replace(' ', '_'))}"
        if enwiki
        else None,
    }

    if not image_name:
        return {
            "status": "missing_image",
            "wikidata": wikidata,
            "commons": None,
        }

    time.sleep(sleep_seconds)
    info = commons_imageinfo(session, image_name)
    if not info:
        return {
            "status": "missing_commons_info",
            "wikidata": wikidata,
            "commons": {"file": image_name},
        }

    return {
        "status": "image_found",
        "wikidata": wikidata,
        "commons": {
            "file": image_name,
            "description_url": info.get("descriptionurl"),
            "url": info.get("url"),
            "thumburl": info.get("thumburl"),
            "mime": info.get("mime"),
            "width": info.get("width"),
            "height": info.get("height"),
            "license": extmetadata_value(info, "LicenseShortName"),
            "usage_terms": extmetadata_value(info, "UsageTerms"),
            "artist": extmetadata_value(info, "Artist"),
            "credit": extmetadata_value(info, "Credit"),
            "attribution_required": extmetadata_value(info, "AttributionRequired"),
            "restrictions": extmetadata_value(info, "Restrictions"),
        },
    }


def resolve_player_image_from_entity(
    entity: dict[str, Any] | None,
    commons_info_by_file: dict[str, dict[str, Any]],
    skip_commons_metadata: bool = False,
) -> dict[str, Any]:
    if not entity:
        return {
            "status": "missing_wikidata",
            "wikidata": None,
            "commons": None,
        }

    qid = entity.get("id")
    image_name = claim_value(entity, "P18")
    enwiki = entity.get("sitelinks", {}).get("enwiki", {}).get("title")
    wikidata = {
        "id": qid,
        "label": entity.get("labels", {}).get("en", {}).get("value"),
        "description": entity.get("descriptions", {}).get("en", {}).get("value"),
        "concepturi": f"http://www.wikidata.org/entity/{qid}" if qid else None,
        "enwiki_title": enwiki,
        "enwiki_url": f"https://en.wikipedia.org/wiki/{quote(enwiki.replace(' ', '_'))}"
        if enwiki
        else None,
    }

    if not image_name:
        return {
            "status": "missing_image",
            "wikidata": wikidata,
            "commons": None,
        }

    if skip_commons_metadata:
        redirect_url = f"https://commons.wikimedia.org/wiki/Special:Redirect/file/{quote(image_name)}"
        return {
            "status": "image_found_needs_commons_review",
            "wikidata": wikidata,
            "commons": {
                "file": image_name,
                "description_url": f"https://commons.wikimedia.org/wiki/File:{quote(image_name.replace(' ', '_'))}",
                "url": redirect_url,
                "thumburl": redirect_url,
                "mime": None,
                "width": None,
                "height": None,
                "license": "NEEDS_COMMONS_REVIEW",
                "usage_terms": None,
                "artist": None,
                "credit": None,
                "attribution_required": None,
                "restrictions": "metadata_not_fetched_due_to_api_throttling",
            },
        }

    info = commons_info_by_file.get(image_name)
    if not info:
        return {
            "status": "missing_commons_info",
            "wikidata": wikidata,
            "commons": {"file": image_name},
        }

    return {
        "status": "image_found",
        "wikidata": wikidata,
        "commons": {
            "file": image_name,
            "description_url": info.get("descriptionurl"),
            "url": info.get("url"),
            "thumburl": info.get("thumburl"),
            "mime": info.get("mime"),
            "width": info.get("width"),
            "height": info.get("height"),
            "license": extmetadata_value(info, "LicenseShortName"),
            "usage_terms": extmetadata_value(info, "UsageTerms"),
            "artist": extmetadata_value(info, "Artist"),
            "credit": extmetadata_value(info, "Credit"),
            "attribution_required": extmetadata_value(info, "AttributionRequired"),
            "restrictions": extmetadata_value(info, "Restrictions"),
        },
    }


def download_image(session: requests.Session, url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = session.get(url, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    tournament = TOURNAMENTS[args.tournament]
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    if args.squads_html:
        page_html = args.squads_html.read_text(encoding="utf-8")
    else:
        print("[FETCH] squads page HTML", flush=True)
        page_html = fetch_squads_page_html(session, tournament["wikipedia_squads_url"])
    teams_payload = []

    for team_idx, team in enumerate(args.teams, start=1):
        print(f"[TEAM {team_idx}/{len(args.teams)}] {team}", flush=True)
        html = find_team_table_html(page_html, team)
        if html is None:
            teams_payload.append(
                {
                    "team": team,
                    "status": "missing_section",
                    "players": [],
                    "notes": "No se encontro esta seleccion en la pagina de squads 2022.",
                }
            )
            continue

        time.sleep(args.sleep)
        table = extract_squad_table(html)
        players = table_to_players(team, table)
        attach_wiki_titles(players, extract_player_wiki_titles(html))
        if args.max_per_team:
            players = players[: args.max_per_team]

        titles = [player.enwiki_title for player in players if player.enwiki_title]
        print(f"  [BATCH] enwiki titles: {len(titles)}", flush=True)
        qid_by_title = enwiki_titles_to_qids(session, titles)
        qids = list(dict.fromkeys(qid_by_title.values()))
        print(f"  [BATCH] wikidata ids: {len(qids)}", flush=True)
        entity_by_qid = get_wikidata_entities(session, qids)
        image_names = [
            claim_value(entity, "P18")
            for entity in entity_by_qid.values()
            if claim_value(entity, "P18")
        ]
        print(f"  [BATCH] commons images: {len(image_names)}", flush=True)
        if args.skip_commons_metadata:
            commons_info_by_file = {}
        else:
            commons_info_by_file = commons_imageinfos(session, list(dict.fromkeys(image_names)))

        player_payloads = []
        for player_idx, player in enumerate(players, start=1):
            print(
                f"  [PLAYER {player_idx}/{len(players)}] {player.name}",
                flush=True,
            )
            qid = qid_by_title.get(player.enwiki_title or "")
            entity = entity_by_qid.get(qid or "")
            if entity:
                resolved = resolve_player_image_from_entity(
                    entity,
                    commons_info_by_file,
                    skip_commons_metadata=args.skip_commons_metadata,
                )
            elif not args.search_fallback:
                resolved = {
                    "status": "missing_wikidata",
                    "wikidata": {
                        "enwiki_title": player.enwiki_title,
                        "enwiki_url": (
                            f"https://en.wikipedia.org/wiki/{quote(player.enwiki_title.replace(' ', '_'))}"
                            if player.enwiki_title
                            else None
                        ),
                    }
                    if player.enwiki_title
                    else None,
                    "commons": None,
                }
            else:
                resolved = resolve_player_image(session, player, args.sleep)
            local_image = None
            if args.download_images:
                commons = resolved.get("commons") or {}
                image_url = commons.get("thumburl") or commons.get("url")
                if image_url:
                    suffix = Path(commons.get("file", "image.jpg")).suffix or ".jpg"
                    local_image_path = (
                        args.image_dir
                        / slugify(team)
                        / f"{slugify(player.name)}{suffix.lower()}"
                    )
                    download_image(session, image_url, local_image_path)
                    local_image = str(local_image_path.relative_to(PROJECT_ROOT))

            player_payloads.append(
                {
                    "team": player.team,
                    "number": player.number,
                    "position": player.position,
                    "name": player.name,
                    "date_of_birth": player.date_of_birth,
                    "caps": player.caps,
                    "goals": player.goals,
                    "club": player.club,
                    "resolution": resolved,
                    "local_image": local_image,
                }
            )
            time.sleep(args.sleep)

        teams_payload.append(
            {
                "team": team,
                "status": "ok",
                "players_count": len(player_payloads),
                "image_found_count": sum(
                    1 for item in player_payloads if item["resolution"]["status"] == "image_found"
                ),
                "players": player_payloads,
            }
        )

    return {
        "schema": "phyloface-teams-photo-manifest-v0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tournament": tournament["label"],
        "tournament_key": args.tournament,
        "squad_status": tournament["squad_status"],
        "sources": {
            "wikipedia_squads": tournament["wikipedia_squads_url"],
            "fifa_squads": tournament["fifa_squads_url"],
            "wikidata_api": WIKIDATA_API,
            "commons_api": COMMONS_API,
        },
        "download_images": bool(args.download_images),
        "teams": teams_payload,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye manifiesto de fotos Wikimedia/Wikidata para la vitrina.",
    )
    parser.add_argument(
        "--tournament",
        choices=sorted(TOURNAMENTS),
        default="northamerica2026",
        help="Torneo objetivo. 2026 es el default; 2022 queda como fallback historico.",
    )
    parser.add_argument(
        "--teams",
        nargs="+",
        default=["Argentina", "France", "Spain", "Mexico", "United States", "Canada"],
        help="Selecciones a procesar segun nombres de la pagina Wikipedia en ingles.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Ruta del JSON de manifiesto.",
    )
    parser.add_argument(
        "--squads-html",
        type=Path,
        default=None,
        help="HTML local de la pagina de squads; evita descargar Wikipedia desde Python.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directorio para imagenes descargadas si se usa --download-images.",
    )
    parser.add_argument(
        "--download-images",
        action="store_true",
        help="Descarga thumb/original de Commons. Por defecto solo genera metadata.",
    )
    parser.add_argument(
        "--max-per-team",
        type=int,
        default=None,
        help="Limite de jugadores por seleccion para smoke tests.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Pausa entre requests para ser conservadores con APIs publicas.",
    )
    parser.add_argument(
        "--search-fallback",
        action="store_true",
        help="Si un link de Wikipedia no resuelve a Wikidata, intenta wbsearch individual.",
    )
    parser.add_argument(
        "--skip-commons-metadata",
        action="store_true",
        help=(
            "No consulta Commons imageinfo; usa Special:Redirect/file desde Wikidata P18 "
            "y marca licencia como NEEDS_COMMONS_REVIEW."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tournament = TOURNAMENTS[args.tournament]
    if args.output == DEFAULT_OUTPUT:
        args.output = tournament["default_output"]
    if args.image_dir == DEFAULT_IMAGE_DIR:
        args.image_dir = tournament["default_image_dir"]
    args.output = args.output.resolve()
    args.image_dir = args.image_dir.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(args)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Manifest escrito: {args.output}")
    for team in manifest["teams"]:
        print(
            f"- {team['team']}: {team.get('status')} | "
            f"{team.get('image_found_count', 0)}/{team.get('players_count', 0)} imagenes"
        )


if __name__ == "__main__":
    main()
