"""Browser glue: thin JSON API over wc_model, called from app.js via Pyodide."""
import json
import sys

sys.path.insert(0, "/wc")

import pandas as pd  # noqa: E402
from wc_model import config, data, model, scoreline  # noqa: E402

STATE = None


def retrain():
    global STATE
    STATE = model.train()
    return STATE["asof"]


def load_cached_state():
    """Use ratings.json restored from localStorage instead of retraining."""
    global STATE
    STATE = model.load_state(retrain=False)
    return STATE["asof"]


def predict(home, away, venue="", neutral=None):
    r = model.predict_match(STATE, home, away,
                            venue_country=venue or None, neutral=neutral)
    r["matrix"] = [[float(x) for x in row] for row in r["matrix"]]
    return json.dumps(r, default=float)


def fixtures():
    df = data.load_results()
    played, fx = data.split_played_fixtures(df)
    wc = played[(played["tournament"] == "FIFA World Cup")
                & (played["date"] >= config.WC_START)]
    rows = []
    for _, r in fx.iterrows():
        rows.append({"date": str(r["date"].date()), "home": r["home_team"],
                     "away": r["away_team"], "city": r["city"],
                     "country": r["country"], "neutral": bool(r["neutral"]),
                     "score": None})
    for _, r in wc.iterrows():
        rows.append({"date": str(r["date"].date()), "home": r["home_team"],
                     "away": r["away_team"], "city": r["city"],
                     "country": r["country"], "neutral": bool(r["neutral"]),
                     "score": f"{int(r['home_score'])}-{int(r['away_score'])}"})
    rows.sort(key=lambda x: x["date"])
    return json.dumps(rows)


def teams_table():
    t = data.load_teams().reset_index()
    out = []
    for _, r in t.iterrows():
        name = r["team"]
        f = STATE["form"].get(name, {})
        out.append({
            "team": name, "group": r["group"], "conf": r["confederation"],
            "fifa_rank": int(r["fifa_rank"]),
            "elo": round(STATE["elo"].get(name, config.ELO_START)),
            "strength": round(50 + 15 * STATE["strength"].get(name, 0), 1),
            "form": f.get("wdl", ""), "ppg": round(f.get("ppg", 0), 2),
        })
    out.sort(key=lambda x: -x["strength"])
    return json.dumps(out)


def recalibrate():
    out = model.calibrate(STATE, verbose=False)
    retrain()
    checks = []
    for (h, a, venue, mu_h, mu_a, ph, pdr, pa, _, _) in config.get_anchors():
        mh, ma = model.expected_goals(STATE, h, a, venue_country=venue)
        m = scoreline.score_matrix(mh, ma, STATE["params"]["rho"])
        p = scoreline.outcome_probs(m)
        checks.append({"match": f"{h} vs {a}",
                       "model": f"{mh:.2f}-{ma:.2f}", "target": f"{mu_h}-{mu_a}",
                       "p": [round(x, 2) for x in p],
                       "p_target": [ph, pdr, pa]})
    return json.dumps({"params": out, "checks": checks})


def wc_progress():
    df = data.load_results()
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"] >= config.WC_START)]
    done = int(wc["home_score"].notna().sum())
    return json.dumps({"done": done, "total": int(len(wc))})


def read_file(path):
    p = config.DATA / path
    return p.read_text(encoding="utf-8") if p.exists() else ""
