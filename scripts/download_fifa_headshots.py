#!/usr/bin/env python3
# =========================================
# ID: PHYLOFACE_FIFA_HEADSHOT_DOWNLOAD
# VERSION: v0.1
# =========================================
# FILE: scripts/download_fifa_headshots.py
#
# Descarga las fotos oficiales FIFA listadas en el manifiesto de
# build_fifa_squad_manifest.py. Diseñado para ser GENTIL con el CDN
# (digitalhub.fifa.com) y evitar baneos:
#   - secuencial (sin concurrencia), pacing con jitter entre descargas
#   - pausa larga cada N imágenes
#   - UA + Referer de browser real
#   - backoff escalado ante 429/5xx; cooldown creciente si hay 429 seguidos
#   - REANUDABLE: saltea archivos ya bajados (escritura atómica tmp->rename)
#
# Copyright FIFA/Getty: uso local de investigación/inferencia. NO publicar.
#
# Uso:
#   conda run -n face-sim python scripts/download_fifa_headshots.py
#   ... --limit 30            # smoke
#   ... --width 1024          # re-pide otra resolución sobre photo_base_url
#   ... --base-delay 0.6 --jitter 0.6 --long-pause-every 100 --long-pause 8

import argparse
import json
import os
import random
import re
import sys
import time
import unicodedata

import requests

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_IN = os.path.join(REPO_ROOT, "data", "output", "teams",
                          "manifest_fifa_northamerica2026_official.json")
DEFAULT_OUT = os.path.join(REPO_ROOT, "data", "input", "img", "teams_players",
                           "northamerica2026_fifa_official")
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def slug(text):
    if not text:
        return "unknown"
    t = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    t = re.sub(r"[^a-zA-Z0-9]+", "-", t).strip("-").lower()
    return t or "unknown"


def make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": UA,
        "Accept": "image/avif,image/webp,image/png,image/*,*/*;q=0.8",
        "Referer": "https://www.fifa.com/",
        "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    })
    return s


def photo_url(player, width, quality):
    """Usa photo_url_best del manifiesto, o re-arma sobre la base si se pide otro width."""
    base = player.get("photo_base_url")
    if width and base:
        return (f"{base}?io=transform:fill,aspectratio:1x1,"
                f"width:{width},gravity:top&quality={quality}")
    return player.get("photo_url_best")


def download_one(session, url, dest, timeout, max_retries):
    """Devuelve (ok, status_note). Backoff escalado; 429 = cooldown más largo."""
    last = None
    for attempt in range(max_retries):
        try:
            r = session.get(url, timeout=timeout, stream=True)
            if r.status_code == 429:
                cd = min(15 * (attempt + 1), 90)
                last = f"429 (cooldown {cd}s)"
                time.sleep(cd)
                continue
            if r.status_code in (500, 502, 503, 504):
                last = f"HTTP {r.status_code}"
                time.sleep(min(3.0 * (attempt + 1), 20))
                continue
            r.raise_for_status()
            tmp = dest + ".part"
            n = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        n += len(chunk)
            if n < 1024:  # archivo sospechosamente chico
                os.remove(tmp)
                last = f"too_small ({n}b)"
                time.sleep(2.0)
                continue
            os.replace(tmp, dest)
            return True, f"{n}b"
        except requests.RequestException as e:
            last = str(e)[:80]
            time.sleep(min(3.0 * (attempt + 1), 20))
    return False, last or "unknown"


def main():
    ap = argparse.ArgumentParser(description="Descarga gentil de fotos oficiales FIFA.")
    ap.add_argument("--input", default=DEFAULT_IN)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--width", type=int, default=None,
                    help="re-pide esta resolución sobre la base; default = photo_url_best del manifiesto.")
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--limit", type=int, default=None, help="cap de descargas (smoke).")
    ap.add_argument("--base-delay", type=float, default=0.5)
    ap.add_argument("--jitter", type=float, default=0.5)
    ap.add_argument("--long-pause-every", type=int, default=100)
    ap.add_argument("--long-pause", type=float, default=8.0)
    ap.add_argument("--timeout", type=float, default=40.0)
    ap.add_argument("--max-retries", type=int, default=4)
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        manifest = json.load(f)

    # Aplanar a (team_slug, player, url, dest).
    jobs = []
    for team in manifest.get("teams", []):
        tslug = slug(team.get("team_name"))
        for p in team.get("players", []):
            url = photo_url(p, args.width, args.quality)
            if not url:
                continue
            fname = f"{slug(p.get('name'))}_{p.get('id_player')}.png"
            dest = os.path.join(args.out_dir, tslug, fname)
            jobs.append((team.get("team_name"), p.get("name"), url, dest))

    if args.limit:
        jobs = jobs[:args.limit]

    total = len(jobs)
    done = skipped = failed = 0
    failures = []
    t0 = time.time()
    print(f"[dl] {total} fotos -> {args.out_dir}")
    print(f"[dl] pacing: base={args.base_delay}s +jitter≤{args.jitter}s, "
          f"pausa {args.long_pause}s cada {args.long_pause_every}")

    session = make_session()
    for i, (team_name, pname, url, dest) in enumerate(jobs, 1):
        if os.path.exists(dest):
            skipped += 1
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        ok, note = download_one(session, url, dest, args.timeout, args.max_retries)
        if ok:
            done += 1
        else:
            failed += 1
            failures.append({"team": team_name, "player": pname, "url": url, "error": note})
            print(f"  [{i}/{total}] FALLO {team_name} / {pname}: {note}")

        if i % 50 == 0 or i == total:
            rate = i / max(time.time() - t0, 1e-6)
            print(f"  [{i}/{total}] ok={done} skip={skipped} fail={failed} "
                  f"({rate:.1f}/s)")
        # Pacing anti-baneo (solo si efectivamente descargamos algo).
        if ok:
            time.sleep(args.base_delay + random.uniform(0, args.jitter))
            if args.long_pause_every and i % args.long_pause_every == 0:
                time.sleep(args.long_pause)

    # Reporte.
    report = {
        "out_dir": args.out_dir,
        "total": total, "downloaded": done, "skipped": skipped, "failed": failed,
        "elapsed_s": round(time.time() - t0, 1),
        "failures": failures,
    }
    rep_path = os.path.join(os.path.dirname(args.input),
                            "download_fifa_headshots_report.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[dl] FIN: ok={done} skip={skipped} fail={failed} "
          f"en {report['elapsed_s']}s")
    print(f"[dl] reporte -> {rep_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
