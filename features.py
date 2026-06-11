"""Team features: form, qualification strength, experience, squad depth."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

MAJOR_FINALS = {"FIFA World Cup", "UEFA Euro", "Copa América",
                "African Cup of Nations", "AFC Asian Cup", "Gold Cup",
                "CONCACAF Championship", "Oceania Nations Cup"}


def _team_matches(played: pd.DataFrame, team: str) -> pd.DataFrame:
    m = played[(played["home_team"] == team) | (played["away_team"] == team)].copy()
    is_home = m["home_team"] == team
    m["gf"] = np.where(is_home, m["home_score"], m["away_score"])
    m["ga"] = np.where(is_home, m["away_score"], m["home_score"])
    m["opp"] = np.where(is_home, m["away_team"], m["home_team"])
    m["pts"] = np.select([m["gf"] > m["ga"], m["gf"] == m["ga"]], [3, 1], 0)
    return m


def form_last10(played: pd.DataFrame, team: str, elo: dict[str, float]) -> dict:
    """Last-10 form: ppg, goals, and Elo-expectation-adjusted performance."""
    m = _team_matches(played, team).sort_values("date").tail(10)
    if m.empty:
        return {"games": 0, "ppg": 1.0, "gf": 1.2, "ga": 1.2, "perf": 0.0, "wdl": ""}
    e_team = elo.get(team, config.ELO_START)
    exp = np.array([1 / (1 + 10 ** (-(e_team - elo.get(o, config.ELO_START)) / 400)) for o in m["opp"]])
    actual = np.select([m["gf"] > m["ga"], m["gf"] == m["ga"]], [1.0, 0.5], 0.0)
    wdl = "".join("W" if r == 1 else ("D" if r == 0.5 else "L") for r in actual)
    return {
        "games": len(m),
        "ppg": float(m["pts"].mean()),
        "gf": float(m["gf"].mean()),
        "ga": float(m["ga"].mean()),
        "perf": float((actual - exp).mean()),   # >0 = exceeding Elo expectation
        "wdl": wdl,
    }


def qualification_strength(played: pd.DataFrame, team: str) -> dict:
    """Points/GD per game in 2026 WC qualifying (2023+)."""
    m = _team_matches(played, team)
    q = m[(m["tournament"] == "FIFA World Cup qualification") & (m["date"] >= "2023-01-01")]
    if q.empty:  # hosts skip qualifying: use competitive matches instead
        q = m[(m["date"] >= "2023-01-01") & (m["tournament"] != "Friendly")]
    if q.empty:
        return {"games": 0, "ppg": 1.5, "gd_pg": 0.0}
    return {"games": len(q), "ppg": float(q["pts"].mean()),
            "gd_pg": float((q["gf"] - q["ga"]).mean())}


def experience(played: pd.DataFrame, team: str) -> float:
    """Weighted count of major-finals matches since 2014 (recent count double)."""
    m = _team_matches(played, team)
    m = m[m["tournament"].isin(MAJOR_FINALS) & (m["date"] >= "2014-01-01")]
    recent = (m["date"] >= "2022-01-01").sum()
    return float(len(m) + recent)


def squad_depth() -> dict[str, float] | None:
    """Optional squad market values (team,value_meur). None if file absent."""
    if not config.SQUAD_VALUES_CSV.exists():
        return None
    sv = pd.read_csv(config.SQUAD_VALUES_CSV)
    return dict(zip(sv["team"], np.log1p(sv["value_meur"])))


def zscores(values: dict[str, float]) -> dict[str, float]:
    arr = np.array(list(values.values()), dtype=float)
    mu, sd = arr.mean(), arr.std()
    if sd < 1e-9:
        return {k: 0.0 for k in values}
    return {k: (v - mu) / sd for k, v in values.items()}
