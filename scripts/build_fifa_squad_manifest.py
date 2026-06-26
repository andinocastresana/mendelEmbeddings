#!/usr/bin/env python3
# =========================================
# ID: PHYLOFACE_FIFA_SQUAD_MANIFEST
# VERSION: v0.1
# =========================================
# FILE: scripts/build_fifa_squad_manifest.py
#
# Cosecha las fichas OFICIALES de jugadores del Mundial 2026 desde la API v3 de
# FIFA (api.fifa.com), sin browser. Descubrimiento (sesion 2026-06-26): la pagina
# /teams/<slug>/team-news es una SPA client-rendered, pero por debajo llama a una
# API REST limpia:
#   - lista de equipos -> calendar/matches (104 partidos -> 48 IdTeam distintos)
#   - squad por equipo  -> teams/{IdTeam}/squad?idCompetition=17&idSeason=285023
# idCompetition=17 (Mundial masculino) e idSeason=285023 (2026) son constantes.
# Cada jugador trae PlayerPicture.PictureUrl = base transform de digitalhub, a la
# que se le agrega ?io=transform:fill,aspectratio:1x1,width:N,gravity:top para
# pedir cualquier resolucion (el CDN sirve >=4096).
#
# Salida: data/output/teams/manifest_fifa_northamerica2026_official.json
#
# IMPORTANTE (licencias): las fotos son copyright FIFA/Getty. Uso local de
# investigacion/inferencia. NO publicar/redistribuir. El manifiesto se marca
# publication_ok=false.
#
# Uso:
#   conda run -n face-sim python scripts/build_fifa_squad_manifest.py
#   ... --teams 43948 43922        # subset (smoke), por IdTeam
#   ... --width 4096 --quality 95  # resolucion de la URL "best"
#   ... --language en

import argparse
import datetime as dt
import json
import os
import sys
import time

import requests

COMPETITION_ID = "17"        # FIFA World Cup (men)
SEASON_ID = "285023"         # 2026 (Canada/Mexico/USA)
API = "https://api.fifa.com/api/v3"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUT = os.path.join(REPO_ROOT, "data", "output", "teams",
                           "manifest_fifa_northamerica2026_official.json")


def make_session():
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "application/json"})
    return s


def request_json(session, url, *, retries=4, base_delay=2.0, timeout=30):
    """GET con backoff exponencial ante 429/5xx (patron de los scrapers del repo)."""
    last = None
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}"
                time.sleep(min(base_delay * (attempt + 1), 10.0))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last = str(e)
            time.sleep(min(base_delay * (attempt + 1), 10.0))
    raise RuntimeError(f"fallo GET {url}: {last}")


def loc(value, language):
    """Toma un campo localizado [{Locale, Description}] -> str del idioma pedido o el 1ro."""
    if not isinstance(value, list) or not value:
        return None
    for item in value:
        if (item.get("Locale") or "").lower().startswith(language.lower()):
            return item.get("Description")
    return value[0].get("Description")


def list_teams(session, language):
    """Devuelve [(IdTeam, team_name, IdCountry)] de los 48 participantes via matches."""
    url = (f"{API}/calendar/matches?idCompetition={COMPETITION_ID}"
           f"&idSeason={SEASON_ID}&count=500&language={language}")
    data = request_json(session, url)
    teams = {}
    for match in data.get("Results", []):
        for side in ("Home", "Away"):
            t = match.get(side)
            if t and t.get("IdTeam"):
                teams[t["IdTeam"]] = (loc(t.get("TeamName"), language),
                                      t.get("IdCountry"))
    return [(tid, nm, cc) for tid, (nm, cc) in teams.items()]


def best_photo_url(base, width, quality):
    if not base:
        return None
    return (f"{base}?io=transform:fill,aspectratio:1x1,"
            f"width:{width},gravity:top&quality={quality}")


def parse_player(p, language, width, quality):
    pic = p.get("PlayerPicture") or {}
    base = pic.get("PictureUrl")
    return {
        "id_player": p.get("IdPlayer"),
        "name": loc(p.get("PlayerName"), language),
        "short_name": loc(p.get("ShortName"), language),
        "jersey_number": p.get("JerseyNum"),
        "position": loc(p.get("PositionLocalized"), language),
        "real_position": loc(p.get("RealPositionLocalized"), language),
        "position_code": p.get("Position"),
        "birth_date": (p.get("BirthDate") or "")[:10] or None,
        "height_cm": p.get("Height"),
        "weight_kg": p.get("Weight"),
        "preferred_foot": p.get("PreferredFoot"),
        "country": p.get("IdCountry"),
        "photo_uuid": pic.get("Id"),
        "photo_base_url": base,
        "photo_url_best": best_photo_url(base, width, quality),
        "has_photo": bool(base),
    }


def get_squad(session, id_team, language, width, quality):
    url = (f"{API}/teams/{id_team}/squad?idCompetition={COMPETITION_ID}"
           f"&idSeason={SEASON_ID}&language={language}")
    data = request_json(session, url)
    players = [parse_player(p, language, width, quality)
               for p in data.get("Players", [])]
    return loc(data.get("TeamName"), language), data.get("IdCountry"), players


def main():
    ap = argparse.ArgumentParser(description="Cosecha fichas oficiales FIFA WC2026.")
    ap.add_argument("--teams", nargs="*", help="subset de IdTeam (smoke). Default: todos.")
    ap.add_argument("--language", default="es")
    ap.add_argument("--width", type=int, default=2048, help="width de la URL best (CDN >=4096).")
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--delay", type=float, default=0.5, help="pausa entre equipos (s).")
    ap.add_argument("--output", default=DEFAULT_OUT)
    args = ap.parse_args()

    session = make_session()

    print(f"[fifa] listando equipos (comp {COMPETITION_ID} / season {SEASON_ID})...")
    all_teams = list_teams(session, args.language)
    print(f"[fifa] {len(all_teams)} equipos participantes")
    if args.teams:
        wanted = set(args.teams)
        all_teams = [t for t in all_teams if t[0] in wanted]
        print(f"[fifa] subset -> {len(all_teams)} equipos: {[t[1] for t in all_teams]}")

    teams_out = []
    total_players = total_photos = 0
    for i, (id_team, name_hint, _cc) in enumerate(sorted(all_teams, key=lambda x: (x[1] or "")), 1):
        try:
            team_name, country, players = get_squad(
                session, id_team, args.language, args.width, args.quality)
        except RuntimeError as e:
            print(f"  [{i}/{len(all_teams)}] {name_hint} ({id_team}) FALLO: {e}")
            teams_out.append({"id_team": id_team, "team_name": name_hint,
                              "error": str(e), "players": []})
            continue
        n_photo = sum(1 for p in players if p["has_photo"])
        total_players += len(players)
        total_photos += n_photo
        print(f"  [{i}/{len(all_teams)}] {team_name or name_hint} ({id_team}): "
              f"{len(players)} jugadores, {n_photo} con foto")
        teams_out.append({
            "id_team": id_team,
            "team_name": team_name or name_hint,
            "country": country,
            "players_count": len(players),
            "photos_count": n_photo,
            "players": players,
        })
        time.sleep(args.delay)

    manifest = {
        "schema": "phyloface-fifa-official-headshot-manifest-v0.1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": "fifa-official-api-v3",
        "source_endpoints": {
            "teams": f"{API}/calendar/matches?idCompetition={COMPETITION_ID}&idSeason={SEASON_ID}",
            "squad": f"{API}/teams/{{IdTeam}}/squad?idCompetition={COMPETITION_ID}&idSeason={SEASON_ID}",
        },
        "competition_id": COMPETITION_ID,
        "season_id": SEASON_ID,
        "language": args.language,
        "photo_transform": {
            "aspectratio": "1x1", "gravity": "top",
            "width": args.width, "quality": args.quality,
            "note": "best url construida sobre PlayerPicture.PictureUrl; el CDN sirve cualquier width (>=4096).",
        },
        "license_status": "UNREVIEWED_COPYRIGHT_FIFA_GETTY",
        "publication_ok": False,
        "license_note": "Fotos oficiales FIFA/Getty. Uso local de investigacion/inferencia. NO publicar ni redistribuir.",
        "teams_count": len(teams_out),
        "players_count": total_players,
        "photos_count": total_photos,
        "teams": teams_out,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n[fifa] OK -> {args.output}")
    print(f"[fifa] equipos={len(teams_out)} jugadores={total_players} con_foto={total_photos}")
    if total_players:
        print(f"[fifa] cobertura fotos: {100*total_photos/total_players:.1f}%")


if __name__ == "__main__":
    sys.exit(main())
