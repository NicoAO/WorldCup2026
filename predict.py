"""Predict any WC2026 matchup with exact-score probabilities.

Usage:
  python predict.py "Mexico" "South Africa" --venue Mexico
  python predict.py "Brazil" "Morocco"                 (neutral venue)
  python predict.py --day 2026-06-11                   (all fixtures that day)
  python predict.py --day all                          (every remaining fixture)
Options:
  --retrain      refit ratings before predicting
  --no-market    pure model, ignore sportsbook odds
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from wc_model import config, model

sys.stdout.reconfigure(encoding="utf-8")


def fmt_pct(p: float) -> str:
    return f"{100 * p:.0f}%"


def print_report(r: dict, show_matrix: bool = True) -> None:
    h, a = r["home"], r["away"]
    print("=" * 64)
    print(f"{h}  vs  {a}")
    print("=" * 64)
    eh, ea = r["elo"]
    print(f"Elo: {eh:.0f} vs {ea:.0f}   |   strength index: "
          f"{_strength(r['strength'][0])} vs {_strength(r['strength'][1])}")
    fh, fa = r["form"]
    if fh and fa:
        print(f"Form (last {fh['games']}): {h} {fh['wdl']} ({fh['ppg']:.2f} ppg) | "
              f"{a} {fa['wdl']} ({fa['ppg']:.2f} ppg)")
    print(f"\nProjected score: {h} {r['mu_home']:.2f} - {r['mu_away']:.2f} {a}"
          f"   (total {r['mu_home'] + r['mu_away']:.2f})")
    mh, md, ma = r["model_1x2"]
    print(f"Model 1X2:  {h} {fmt_pct(mh)} | draw {fmt_pct(md)} | {a} {fmt_pct(ma)}")
    if r["market_1x2"]:
        kh, kd, ka = r["market_1x2"]
        print(f"Market 1X2: {h} {fmt_pct(kh)} | draw {fmt_pct(kd)} | {a} {fmt_pct(ka)} (no-vig)")
    f = r["final"]
    print(f"FINAL 1X2:  {h} {fmt_pct(f['p_home'])} | draw {fmt_pct(f['p_draw'])} | "
          f"{a} {fmt_pct(f['p_away'])}")
    print(f"Over 2.5: {fmt_pct(f['over_2_5'])}   BTTS: {fmt_pct(f['btts'])}")
    print("\nMost likely scorelines:")
    for s, p in r["top_scores"]:
        print(f"  {s:>5}  {100 * p:5.1f}%  {'█' * int(round(100 * p))}")
    if show_matrix:
        m = r["matrix"]
        n = 6
        print(f"\nExact-score probability matrix (rows = {h} goals, cols = {a} goals, %):")
        print("      " + "".join(f"{j:>6}" for j in range(n)))
        for i in range(n):
            print(f"  {i:>3} " + "".join(f"{100 * m[i, j]:6.1f}" for j in range(n)))
    print()


def _strength(s) -> str:
    if s is None:
        return "n/a"
    return f"{50 + 15 * s:.0f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("home", nargs="?")
    ap.add_argument("away", nargs="?")
    ap.add_argument("--venue", default=None, help="venue country (host edge if home team hosts)")
    ap.add_argument("--neutral", action="store_true")
    ap.add_argument("--day", default=None, help="YYYY-MM-DD or 'all'")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--no-market", action="store_true")
    ap.add_argument("--no-matrix", action="store_true")
    args = ap.parse_args()

    state = model.load_state(retrain=args.retrain)
    print(f"[ratings as of {state['asof']}]\n")

    if args.day:
        from wc_model import data
        df = data.load_results()
        _, fixtures = data.split_played_fixtures(df)
        if args.day != "all":
            fixtures = fixtures[fixtures["date"] == args.day]
        if fixtures.empty:
            print(f"No unplayed WC fixtures found for {args.day}")
            return
        for _, fx in fixtures.iterrows():
            r = model.predict_match(state, fx["home_team"], fx["away_team"],
                                    venue_country=fx["country"],
                                    neutral=bool(fx["neutral"]),
                                    use_market=not args.no_market)
            print(f"--- {fx['date'].date()}  {fx['city']}, {fx['country']} ---")
            print_report(r, show_matrix=not args.no_matrix)
        return

    if not (args.home and args.away):
        ap.error("provide HOME AWAY team names, or --day")
    neutral = True if args.neutral else None
    r = model.predict_match(state, args.home, args.away,
                            venue_country=args.venue, neutral=neutral,
                            use_market=not args.no_market)
    print_report(r, show_matrix=not args.no_matrix)


if __name__ == "__main__":
    main()
