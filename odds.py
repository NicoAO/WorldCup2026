"""Sportsbook odds: The Odds API client + manual CSV fallback.

Get a free API key at https://the-odds-api.com (500 requests/month) and set:
    $env:ODDS_API_KEY = "yourkey"        (PowerShell)
Without a key, you can still feed odds via data/manual_odds.csv:
    home,away,odds_home,odds_draw,odds_away   (decimal odds)
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

import pandas as pd

from . import config

SPORT_KEYS = ["soccer_fifa_world_cup", "soccer_fifa_world_cup_2026"]
CACHE_TTL = 6 * 3600

# The Odds API team names -> results.csv names
NAME_MAP = {
    "USA": "United States", "Korea Republic": "South Korea",
    "Czechia": "Czech Republic", "Türkiye": "Turkey", "Cabo Verde": "Cape Verde",
    "Côte d'Ivoire": "Ivory Coast", "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Congo DR": "DR Congo", "IR Iran": "Iran",
}


def _norm(name: str) -> str:
    return NAME_MAP.get(name, name)


def fetch_api_odds(force: bool = False) -> list | None:
    """Fetch h2h odds for all WC matches; cached in data/odds_cache.json."""
    if config.ODDS_CACHE.exists() and not force:
        cached = json.loads(config.ODDS_CACHE.read_text(encoding="utf-8"))
        if time.time() - cached.get("fetched_at", 0) < CACHE_TTL:
            return cached["events"]
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        return None
    for sport in SPORT_KEYS:
        url = (f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
               f"?regions=eu,us&markets=h2h&oddsFormat=decimal&apiKey={key}")
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                events = json.loads(r.read())
            if events:
                config.ODDS_CACHE.write_text(
                    json.dumps({"fetched_at": time.time(), "events": events}),
                    encoding="utf-8")
                return events
        except Exception:
            continue
    return None


def _devig(oh: float, od: float, oa: float) -> tuple[float, float, float]:
    """Decimal odds -> no-vig probabilities (proportional method)."""
    inv = (1 / oh, 1 / od, 1 / oa)
    s = sum(inv)
    return inv[0] / s, inv[1] / s, inv[2] / s


def market_probs(home: str, away: str) -> tuple[float, float, float] | None:
    """No-vig 1X2 market probabilities for a fixture, or None if unavailable."""
    events = fetch_api_odds()
    if events:
        for ev in events:
            eh, ea = _norm(ev.get("home_team", "")), _norm(ev.get("away_team", ""))
            if {eh, ea} != {home, away}:
                continue
            quotes = []
            for bk in ev.get("bookmakers", []):
                for mk in bk.get("markets", []):
                    if mk["key"] != "h2h":
                        continue
                    prices = {_norm(o["name"]): o["price"] for o in mk["outcomes"]}
                    if home in prices and away in prices and "Draw" in prices:
                        quotes.append((prices[home], prices["Draw"], prices[away]))
            if quotes:
                q = pd.DataFrame(quotes, columns=["h", "d", "a"]).median()
                p = _devig(q["h"], q["d"], q["a"])
                return p if eh == home else (p[2], p[1], p[0])
    if config.MANUAL_ODDS_CSV.exists():
        mo = pd.read_csv(config.MANUAL_ODDS_CSV)
        row = mo[(mo["home"] == home) & (mo["away"] == away)]
        if not row.empty:
            r = row.iloc[-1]
            return _devig(r["odds_home"], r["odds_draw"], r["odds_away"])
        row = mo[(mo["home"] == away) & (mo["away"] == home)]
        if not row.empty:
            r = row.iloc[-1]
            p = _devig(r["odds_home"], r["odds_draw"], r["odds_away"])
            return p[2], p[1], p[0]
    return None
