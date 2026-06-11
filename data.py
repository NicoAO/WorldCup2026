"""Loading and preparing match data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def load_results() -> pd.DataFrame:
    df = pd.read_csv(config.RESULTS_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df["neutral"] = df["neutral"].astype(bool)
    return df


def split_played_fixtures(df: pd.DataFrame):
    """Split into played matches and upcoming (unscored) WC fixtures."""
    played = df[df["home_score"].notna()].copy()
    played["home_score"] = played["home_score"].astype(int)
    played["away_score"] = played["away_score"].astype(int)
    fixtures = df[df["home_score"].isna() & (df["tournament"] == "FIFA World Cup")].copy()
    return played, fixtures


def match_weights(df: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    """Importance x exponential time-decay weight per match."""
    imp = df["tournament"].map(config.IMPORTANCE).fillna(config.DEFAULT_IMPORTANCE)
    is_wc26 = (df["tournament"] == "FIFA World Cup") & (df["date"] >= pd.Timestamp(config.WC_START))
    imp = imp.where(~is_wc26, config.WC_BOOST)
    days = (asof - df["date"]).dt.days.clip(lower=0)
    decay = 0.5 ** (days / config.HALF_LIFE_DAYS)
    return imp * decay


def soft_cap_goals(g: pd.Series) -> pd.Series:
    """Blowout damping: goals above GOAL_CAP only count 25% (xG-style smoothing)."""
    cap = config.GOAL_CAP
    return np.minimum(g, cap) + 0.25 * np.maximum(g - cap, 0)


def apply_xg_blend(long_df: pd.DataFrame) -> pd.DataFrame:
    """If data/xg.csv exists (date,team,opponent,xg), blend actual goals with xG."""
    if not config.XG_CSV.exists():
        return long_df
    xg = pd.read_csv(config.XG_CSV)
    xg["date"] = pd.to_datetime(xg["date"])
    merged = long_df.merge(xg, on=["date", "team", "opponent"], how="left")
    has = merged["xg"].notna()
    merged.loc[has, "goals_t"] = (
        (1 - config.XG_BLEND) * merged.loc[has, "goals_t"] + config.XG_BLEND * merged.loc[has, "xg"]
    )
    return merged.drop(columns=["xg"])


def to_long(played: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """One row per team-match with training goals, weights, home indicator."""
    w = match_weights(played, asof)
    base = played.assign(weight=w)
    home = base.rename(columns={"home_team": "team", "away_team": "opponent",
                                "home_score": "gf", "away_score": "ga"})
    home["is_home"] = ~base["neutral"]
    away = base.rename(columns={"away_team": "team", "home_team": "opponent",
                                "away_score": "gf", "home_score": "ga"})
    away["is_home"] = False
    cols = ["date", "team", "opponent", "gf", "ga", "is_home", "weight", "tournament"]
    long_df = pd.concat([home[cols], away[cols]], ignore_index=True)
    long_df["goals_t"] = soft_cap_goals(long_df["gf"])
    long_df = apply_xg_blend(long_df)
    return long_df


def load_teams() -> pd.DataFrame:
    t = pd.read_csv(config.TEAMS_CSV)
    t["is_host"] = t["is_host"].astype(bool)
    return t.set_index("team")
