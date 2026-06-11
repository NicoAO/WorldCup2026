"""Rating engines: Elo, weighted SRS, attack/defense Poisson strengths."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------- Elo
def compute_elo(played: pd.DataFrame) -> dict[str, float]:
    """World Football Elo (eloratings.net conventions) over full history."""
    elo: dict[str, float] = {}

    def get(t):
        return elo.get(t, config.ELO_START)

    k_map = config.ELO_K
    rows = played[["home_team", "away_team", "home_score", "away_score",
                   "neutral", "tournament"]].itertuples(index=False)
    for h, a, hs, as_, neutral, tourn in rows:
        rh, ra = get(h), get(a)
        k = k_map.get(tourn, config.ELO_K_DEFAULT)
        dr = rh - ra + (0.0 if neutral else config.ELO_HOME_ADV)
        we = 1.0 / (1.0 + 10 ** (-dr / 400.0))
        res = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        gd = abs(hs - as_)
        if gd <= 1:
            g = 1.0
        elif gd == 2:
            g = 1.5
        else:
            g = (11.0 + gd) / 8.0
        delta = k * g * (res - we)
        elo[h] = rh + delta
        elo[a] = ra - delta
    return elo


# ---------------------------------------------------------------- SRS
def compute_srs(played: pd.DataFrame, asof: pd.Timestamp,
                window_years: float = 4.0) -> dict[str, float]:
    """Weighted Simple Rating System on capped goal difference.

    rating_i = weighted mean over games of (adj_gd + rating_opponent)
    """
    from .data import match_weights

    cutoff = asof - pd.Timedelta(days=int(365.25 * window_years))
    df = played[played["date"] >= cutoff].copy()
    df["w"] = match_weights(df, asof)
    gd = (df["home_score"] - df["away_score"]).clip(-config.SRS_GD_CAP, config.SRS_GD_CAP)
    gd = gd - np.where(df["neutral"], 0.0, config.SRS_HOME_ADV)

    g = pd.concat([
        pd.DataFrame({"team": df["home_team"], "opp": df["away_team"], "gd": gd, "w": df["w"]}),
        pd.DataFrame({"team": df["away_team"], "opp": df["home_team"], "gd": -gd, "w": df["w"]}),
    ], ignore_index=True)

    teams = pd.Index(sorted(set(g["team"])))
    idx = {t: i for i, t in enumerate(teams)}
    ti = g["team"].map(idx).to_numpy().astype(np.intp)   # intp: wasm32-safe for bincount
    oi = g["opp"].map(idx).to_numpy().astype(np.intp)
    w = g["w"].to_numpy()
    gdv = g["gd"].to_numpy()
    wsum = np.bincount(ti, weights=w, minlength=len(teams))
    base = np.bincount(ti, weights=w * gdv, minlength=len(teams)) / np.maximum(wsum, 1e-9)

    r = base.copy()
    for _ in range(200):
        opp_avg = np.bincount(ti, weights=w * r[oi], minlength=len(teams)) / np.maximum(wsum, 1e-9)
        r_new = base + opp_avg
        r_new -= r_new.mean()
        if np.abs(r_new - r).max() < 1e-7:
            r = r_new
            break
        r = 0.5 * r + 0.5 * r_new
    return dict(zip(teams, r))


# ------------------------------------------------- attack/defense Poisson
def fit_attack_defense(long_df: pd.DataFrame, elo: dict[str, float],
                       min_games: int = 5, sweeps: int = 40):
    """Weighted Poisson fit:  log mu = c + home*h + att_team - def_opponent.

    Multiplicative (Maher-style) alternating updates, then Elo-prior shrinkage
    for teams with little data.
    """
    df = long_df.copy()
    teams = pd.Index(sorted(set(df["team"]) | set(df["opponent"])))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    ti = df["team"].map(idx).to_numpy().astype(np.intp)  # intp: wasm32-safe for bincount
    oi = df["opponent"].map(idx).to_numpy().astype(np.intp)
    w = df["weight"].to_numpy()
    y = df["goals_t"].to_numpy(dtype=float)
    home = df["is_home"].to_numpy(dtype=float)

    att = np.zeros(n)
    dfn = np.zeros(n)
    c = np.log(max(np.average(y, weights=w), 1e-6))
    h = 0.25

    for _ in range(sweeps):
        mu = np.exp(c + h * home + att[ti] - dfn[oi])
        # attack update
        num = np.bincount(ti, weights=w * y, minlength=n)
        den = np.bincount(ti, weights=w * mu, minlength=n)
        att += np.log(np.maximum(num, 1e-9) / np.maximum(den, 1e-9))
        att -= att.mean()
        mu = np.exp(c + h * home + att[ti] - dfn[oi])
        # defense update (more conceded -> lower def)
        num = np.bincount(oi, weights=w * y, minlength=n)
        den = np.bincount(oi, weights=w * mu, minlength=n)
        dfn -= np.log(np.maximum(num, 1e-9) / np.maximum(den, 1e-9))
        dfn -= dfn.mean()
        # intercept + home advantage
        mu = np.exp(c + h * home + att[ti] - dfn[oi])
        c += np.log(np.sum(w * y) / np.sum(w * mu))
        mask = home > 0
        if mask.any():
            mu = np.exp(c + h * home + att[ti] - dfn[oi])
            ratio = np.sum(w[mask] * y[mask]) / max(np.sum(w[mask] * mu[mask]), 1e-9)
            h += np.log(max(ratio, 1e-9))

    # Elo-prior shrinkage: teams with few weighted games drift to Elo-implied level
    elo_arr = np.array([elo.get(t, config.ELO_START) for t in teams])
    z = (elo_arr - elo_arr.mean()) / 400.0
    att_prior, def_prior = 0.45 * z, 0.45 * z
    n_eff = np.bincount(ti, weights=w, minlength=n)
    lam = n_eff / (n_eff + config.SHRINK_GAMES)
    att = lam * att + (1 - lam) * att_prior
    dfn = lam * dfn + (1 - lam) * def_prior

    games = np.bincount(ti, minlength=n)
    return {
        "teams": list(teams),
        "att": dict(zip(teams, att)),
        "def": dict(zip(teams, dfn)),
        "intercept": float(c),
        "home_adv": float(h),
        "games": dict(zip(teams, games.astype(int))),
    }
