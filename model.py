"""Model orchestration: training, calibration, and match prediction."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config, data, features, odds, ratings, scoreline


# ----------------------------------------------------------------- training
def train(asof: str | None = None, save: bool = True) -> dict:
    """Fit all ratings/features and cache them to data/ratings.json."""
    df = data.load_results()
    played, fixtures = data.split_played_fixtures(df)
    asof_ts = pd.Timestamp(asof) if asof else max(played["date"].max(), pd.Timestamp.today().normalize())

    elo = ratings.compute_elo(played)
    srs = ratings.compute_srs(played, asof_ts)

    cutoff = asof_ts - pd.Timedelta(days=int(365.25 * config.TRAIN_CUTOFF_YEARS))
    recent = played[played["date"] >= cutoff]
    long_df = data.to_long(recent, asof_ts)
    ad = ratings.fit_attack_defense(long_df, elo)

    teams = data.load_teams()
    wc_teams = list(teams.index)

    form = {t: features.form_last10(played, t, elo) for t in wc_teams}
    qual = {t: features.qualification_strength(played, t) for t in wc_teams}
    exp_raw = {t: features.experience(played, t) for t in wc_teams}
    depth_raw = features.squad_depth()

    form_z = features.zscores({t: f["perf"] for t, f in form.items()})
    exp_z = features.zscores(exp_raw)
    depth_z = features.zscores({t: depth_raw.get(t, 0.0) for t in wc_teams}) if depth_raw else {t: 0.0 for t in wc_teams}

    # composite strength (reporting + "team strength difference" feature)
    elo_z = features.zscores({t: elo.get(t, config.ELO_START) for t in wc_teams})
    net_z = features.zscores({t: ad["att"].get(t, 0) + ad["def"].get(t, 0) for t in wc_teams})
    srs_z = features.zscores({t: srs.get(t, 0.0) for t in wc_teams})
    fifa_z = features.zscores({t: float(teams.loc[t, "fifa_points"]) for t in wc_teams})
    strength = {t: 0.40 * elo_z[t] + 0.25 * net_z[t] + 0.15 * srs_z[t]
                + 0.10 * fifa_z[t] + 0.05 * form_z[t] + 0.05 * exp_z[t]
                for t in wc_teams}

    state = {
        "asof": str(asof_ts.date()),
        "elo": elo, "srs": srs,
        "att": ad["att"], "def": ad["def"],
        "intercept": ad["intercept"], "home_adv_fit": ad["home_adv"],
        "games": ad["games"],
        "form": form, "qual": qual,
        "form_z": form_z, "exp_z": exp_z, "depth_z": depth_z,
        "strength": strength,
        "params": load_calibration(),
    }
    if save:
        config.RATINGS_CACHE.write_text(
            json.dumps(state, ensure_ascii=False, default=float), encoding="utf-8")
    return state


def load_state(retrain: bool = False) -> dict:
    if retrain or not config.RATINGS_CACHE.exists():
        return train()
    state = json.loads(config.RATINGS_CACHE.read_text(encoding="utf-8"))
    state["params"] = load_calibration()
    return state


def load_calibration() -> dict:
    params = dict(config.DEFAULTS)
    if config.CALIBRATION_JSON.exists():
        params.update(json.loads(config.CALIBRATION_JSON.read_text(encoding="utf-8")))
    return params


# ----------------------------------------------------------------- predict
def expected_goals(state: dict, home: str, away: str,
                   venue_country: str | None = None,
                   neutral: bool | None = None) -> tuple[float, float]:
    """Model expected goals for each side (before market blending)."""
    p = state["params"]
    teams = data.load_teams()

    def side(team, opp, is_home_side):
        mu = (state["intercept"] + p["goal_level"]
              + state["att"].get(team, 0.0) - state["def"].get(opp, 0.0))
        if team in teams.index:
            conf = teams.loc[team, "confederation"]
            mu += config.TRAVEL_PENALTY.get(conf, -0.02)
            mu += (p["form_beta"] * state["form_z"].get(team, 0.0)
                   + p["exp_beta"] * state["exp_z"].get(team, 0.0)
                   + p["depth_beta"] * state["depth_z"].get(team, 0.0))
        return mu

    home_edge = _has_home_edge(teams, home, venue_country, neutral)
    log_h = side(home, away, True) + (p["hfa_goals"] if home_edge else 0.0)
    log_a = side(away, home, False)
    mu_h, mu_a = float(np.exp(log_h)), float(np.exp(log_a))

    # blend goal difference with the Elo-implied goal difference
    elo_h = state["elo"].get(home, config.ELO_START) + (config.ELO_HOME_ADV if home_edge else 0.0)
    elo_a = state["elo"].get(away, config.ELO_START)
    gd_elo = (elo_h - elo_a) / p["elo_gd_scale"]
    gd = (1 - p["elo_blend"]) * (mu_h - mu_a) + p["elo_blend"] * gd_elo
    total = mu_h + mu_a
    mu_h = float(np.clip((total + gd) / 2, 0.12, 5.5))
    mu_a = float(np.clip(total - mu_h, 0.12, 5.5))
    return mu_h, mu_a


def _has_home_edge(teams, home, venue_country, neutral):
    if neutral is False:
        return True
    if neutral is True:
        return False
    if venue_country and home in teams.index:
        return bool(teams.loc[home, "is_host"]) and venue_country == home
    return False


def predict_match(state: dict, home: str, away: str,
                  venue_country: str | None = None,
                  neutral: bool | None = None,
                  use_market: bool = True) -> dict:
    mu_h, mu_a = expected_goals(state, home, away, venue_country, neutral)
    m = scoreline.score_matrix(mu_h, mu_a, state["params"]["rho"])
    summary = scoreline.market_summary(m)

    market = odds.market_probs(home, away) if use_market else None
    if market is not None:
        m_final = scoreline.blend_with_market(m, market, state["params"]["market_weight"])
    else:
        m_final = m
    final = scoreline.market_summary(m_final)

    return {
        "home": home, "away": away,
        "mu_home": mu_h, "mu_away": mu_a,
        "matrix": m_final,
        "model_1x2": (summary["p_home"], summary["p_draw"], summary["p_away"]),
        "market_1x2": market,
        "final": final,
        "top_scores": scoreline.top_scorelines(m_final),
        "elo": (state["elo"].get(home, config.ELO_START), state["elo"].get(away, config.ELO_START)),
        "strength": (state["strength"].get(home), state["strength"].get(away)),
        "form": (state["form"].get(home), state["form"].get(away)),
    }


# ----------------------------------------------------------------- calibrate
def calibrate(state: dict | None = None, verbose: bool = True) -> dict:
    """Tune global parameters to match the Action Network day-1 anchors."""
    from scipy.optimize import minimize

    if state is None:
        state = load_state()

    def loss(x):
        params = dict(state["params"])
        params.update({"goal_level": x[0], "hfa_goals": x[1],
                       "elo_gd_scale": 150 + 350 / (1 + np.exp(-x[2])),
                       "elo_blend": 0.15 + 0.6 / (1 + np.exp(-x[3]))})
        st = {**state, "params": params}
        tot = 0.0
        for (h, a, venue, mu_h, mu_a, ph, pd_, pa, _, _) in config.get_anchors():
            mh, ma = expected_goals(st, h, a, venue_country=venue)
            mat = scoreline.score_matrix(mh, ma, params["rho"])
            p = scoreline.outcome_probs(mat)
            tot += (mh - mu_h) ** 2 + (ma - mu_a) ** 2
            tot += 4 * ((p[0] - ph) ** 2 + (p[1] - pd_) ** 2 + (p[2] - pa) ** 2)
        return tot

    d = config.DEFAULTS
    x0 = [d["goal_level"], d["hfa_goals"],
          np.log((d["elo_gd_scale"] - 150) / (500 - d["elo_gd_scale"])),
          np.log((d["elo_blend"] - 0.15) / (0.75 - d["elo_blend"]))]
    res = minimize(loss, x0, method="Nelder-Mead",
                   options={"maxiter": 600, "xatol": 1e-4, "fatol": 1e-6})
    out = {"goal_level": float(res.x[0]), "hfa_goals": float(res.x[1]),
           "elo_gd_scale": float(150 + 350 / (1 + np.exp(-res.x[2]))),
           "elo_blend": float(0.15 + 0.6 / (1 + np.exp(-res.x[3])))}
    config.CALIBRATION_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    state["params"].update(out)
    if verbose:
        print(f"calibration loss: {res.fun:.5f}")
        print(json.dumps(out, indent=2))
    return out


# ----------------------------------------------------------------- fixtures
def fixtures_on(day: str) -> pd.DataFrame:
    df = data.load_results()
    _, fixtures = data.split_played_fixtures(df)
    return fixtures[fixtures["date"] == pd.Timestamp(day)]
