"""Refresh results (new World Cup scores) and retrain the model.

World Cup 2026 matches enter training with weight WC_BOOST (8x a friendly)
and near-zero time decay, so they dominate the ratings as the tournament runs.

Usage:  python update.py [--no-download]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

import pandas as pd

from wc_model import config, model

sys.stdout.reconfigure(encoding="utf-8")


def download_results() -> bool:
    tmp = config.RESULTS_CSV.with_suffix(".new")
    cmd = ["curl.exe" if shutil.which("curl.exe") else "curl",
           "-sL", "--max-time", "120", "-o", str(tmp), config.RESULTS_URL]
    try:
        subprocess.run(cmd, check=True)
        head = tmp.read_text(encoding="utf-8", errors="ignore")[:100]
        if not head.startswith("date,home_team"):
            print("Download looks wrong, keeping existing results.csv")
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(config.RESULTS_CSV)
        return True
    except Exception as e:
        print(f"Download failed ({e}); keeping existing results.csv")
        tmp.unlink(missing_ok=True)
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    if not args.no_download:
        print(f"Downloading latest results from {config.RESULTS_URL} ...")
        download_results()

    df = pd.read_csv(config.RESULTS_CSV)
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"] >= config.WC_START)]
    done = wc[wc["home_score"].notna()]
    print(f"\nWorld Cup 2026: {len(done)}/{len(wc)} matches have results.")
    if not done.empty:
        print("Completed WC matches (weight 8x in training):")
        for _, r in done.iterrows():
            print(f"  {r['date']}  {r['home_team']} {int(r['home_score'])}"
                  f"-{int(r['away_score'])} {r['away_team']}")

    print("\nRetraining ratings with updated data...")
    state = model.train()
    print(f"Done. Ratings as of {state['asof']}.")
    print("Run:  python predict.py --day <YYYY-MM-DD>   for the next slate.")


if __name__ == "__main__":
    main()
