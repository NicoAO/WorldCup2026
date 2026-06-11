"""Fit all ratings, calibrate to Action Network anchors, and cache.

Usage:  python train.py [--no-calibrate]
"""
from __future__ import annotations

import argparse
import sys

from wc_model import config, model, scoreline

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-calibrate", action="store_true")
    args = ap.parse_args()

    print("Training ratings (Elo + SRS + attack/defense Poisson)...")
    state = model.train()
    print(f"  trained as of {state['asof']}  ({len(state['elo'])} teams rated)")

    # sanity check vs Action Network published Elo
    print("\nElo check vs Action Network (Jun 10):")
    for (h, a, _, _, _, _, _, _, an_h, an_a) in config.get_anchors():
        print(f"  {h:<16} model {state['elo'].get(h, 0):.0f} vs AN {an_h}   | "
              f"{a:<14} model {state['elo'].get(a, 0):.0f} vs AN {an_a}")

    if not args.no_calibrate:
        print("\nCalibrating global parameters to Action Network day-1 projections...")
        model.calibrate(state)
        # persist calibrated params inside the cache too
        state = model.train()

    print("\nPost-calibration check on anchor matches:")
    for (h, a, venue, mu_h, mu_a, ph, pdr, pa, _, _) in config.get_anchors():
        mh, ma = model.expected_goals(state, h, a, venue_country=venue)
        m = scoreline.score_matrix(mh, ma, state["params"]["rho"])
        p = scoreline.outcome_probs(m)
        print(f"  {h} vs {a}:")
        print(f"    model {mh:.2f}-{ma:.2f} (target {mu_h:.1f}-{mu_a:.1f}) | "
              f"1X2 {p[0]:.0%}/{p[1]:.0%}/{p[2]:.0%} (target {ph:.0%}/{pdr:.0%}/{pa:.0%})")

    top = sorted(state["strength"].items(), key=lambda kv: -kv[1])[:12]
    print("\nTop 12 composite strength (Elo/att-def/SRS/FIFA/form/experience):")
    for i, (t, s) in enumerate(top, 1):
        print(f"  {i:>2}. {t:<16} {50 + 15 * s:5.1f}")


if __name__ == "__main__":
    main()
