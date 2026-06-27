#!/usr/bin/env python3
# =========================================
# ID: PHYLOFACE_GEO_TEAM_RESOLVE
# VERSION: v0.1
# =========================================
# FILE: scripts/geo_team_resolve.py
#
# Helper compartido por los scripts de visualización geo/colonial (team scatter,
# player scatter, mapa). Resuelve cada SELECCIÓN del payload de vitrina FIFA
# (nombres en español) a su registro geo (capital lat/lon + colonizador + idiomas)
# de world_capitals_distance.json, y expone utilidades comunes.

import json
import os
import re
import unicodedata

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_GEO = os.path.join(REPO_ROOT, "data", "output", "geo",
                           "world_capitals_distance.json")

# 48 selecciones del Mundial 2026 (nombre español del payload FIFA) -> ISO3.
# Escocia + Inglaterra -> GBR (capital soberana = Londres).
TEAM_ES_TO_CCA3 = {
    "alemania": "DEU", "arabia saudi": "SAU", "argelia": "DZA", "argentina": "ARG",
    "australia": "AUS", "austria": "AUT", "bosnia y herzegovina": "BIH",
    "brasil": "BRA", "belgica": "BEL", "canada": "CAN", "catar": "QAT",
    "chequia": "CZE", "colombia": "COL", "costa de marfil": "CIV", "croacia": "HRV",
    "curazao": "CUW", "ee uu": "USA", "ecuador": "ECU", "egipto": "EGY",
    "escocia": "GBR", "espana": "ESP", "francia": "FRA", "ghana": "GHA",
    "haiti": "HTI", "inglaterra": "GBR", "irak": "IRQ", "islas de cabo verde": "CPV",
    "japon": "JPN", "jordania": "JOR", "marruecos": "MAR", "mexico": "MEX",
    "noruega": "NOR", "nueva zelanda": "NZL", "panama": "PAN", "paraguay": "PRY",
    "paises bajos": "NLD", "portugal": "PRT", "rd congo": "COD", "ri de iran": "IRN",
    "republica de corea": "KOR", "senegal": "SEN", "sudafrica": "ZAF",
    "suecia": "SWE", "suiza": "CHE", "turquia": "TUR", "tunez": "TUN",
    "uruguay": "URY", "uzbekistan": "UZB",
}

NON_LINK_COL = {None, "Never colonized", "Colonizer power", "Multiple colonizers"}


def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def make_resolver(geo_path=DEFAULT_GEO):
    """Devuelve (resolve, by_cca3). resolve(team_es) -> registro geo o None."""
    d = json.load(open(geo_path, encoding="utf-8"))
    by_cca3 = {c["cca3"]: c for c in d["countries"] if c.get("cca3")}
    by_name = {}
    for c in d["countries"]:
        for nm in (c["country"], c.get("country_raw")):
            if nm:
                by_name.setdefault(norm(nm), c)

    def resolve(team_name):
        cca3 = TEAM_ES_TO_CCA3.get(norm(team_name))
        if cca3 and cca3 in by_cca3:
            return by_cca3[cca3]
        return by_name.get(norm(team_name))   # fallback nombre directo (inglés)

    return resolve, by_cca3


def haversine(a, b):
    R = 6371.0
    la1, lo1, la2, lo2 = map(np.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    h = (np.sin((la2 - la1) / 2) ** 2
         + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)
    return float(2 * R * np.arcsin(np.sqrt(h)))


def colonial_link(a, b):
    """colonia-colonia (mismo colonizador) OR colonia-colonizador."""
    ca, cb = a.get("last_colonizer"), b.get("last_colonizer")
    if ca and ca == cb and ca not in NON_LINK_COL:
        return True
    if ca and ca not in NON_LINK_COL and norm(ca) == norm(b["country"]):
        return True
    if cb and cb not in NON_LINK_COL and norm(cb) == norm(a["country"]):
        return True
    return False


def share_language(a, b):
    return bool(set(a["languages"]) & set(b["languages"]))


def partial_corr(x, y, z):
    """r(x,y | z) de primer orden."""
    x, y, z = map(np.asarray, (x, y, z))
    rxy = np.corrcoef(x, y)[0, 1]
    rxz = np.corrcoef(x, z)[0, 1]
    ryz = np.corrcoef(y, z)[0, 1]
    denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return (rxy - rxz * ryz) / denom if denom > 1e-9 else float("nan")
