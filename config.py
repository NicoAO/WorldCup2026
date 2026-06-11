"""Global configuration for the WC2026 prediction model."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS_CSV = DATA / "results.csv"
TEAMS_CSV = DATA / "teams.csv"
XG_CSV = DATA / "xg.csv"                    # optional: date,team,opponent,xg
SQUAD_VALUES_CSV = DATA / "squad_values.csv"  # optional: team,value_meur
MANUAL_ODDS_CSV = DATA / "manual_odds.csv"  # optional: home,away,odds_home,odds_draw,odds_away
ODDS_CACHE = DATA / "odds_cache.json"
RATINGS_CACHE = DATA / "ratings.json"
CALIBRATION_JSON = DATA / "calibration.json"

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

WC_START = "2026-06-11"           # tournament start: results on/after this get WC_BOOST
TRAIN_CUTOFF_YEARS = 8            # attack/defense + SRS training window
HALF_LIFE_DAYS = 730.0            # time-decay half-life (2 years)

# Match-importance weights (multiplied by time decay).  WC 2026 games dominate.
IMPORTANCE = {
    "FIFA World Cup": 4.0,
    "FIFA World Cup qualification": 3.0,
    "UEFA Euro": 3.0,
    "Copa América": 3.0,
    "African Cup of Nations": 3.0,
    "AFC Asian Cup": 3.0,
    "Gold Cup": 2.5,
    "CONCACAF Championship": 2.5,
    "Oceania Nations Cup": 2.5,
    "UEFA Nations League": 2.0,
    "CONCACAF Nations League": 1.5,
    "UEFA Euro qualification": 2.0,
    "African Cup of Nations qualification": 2.0,
    "AFC Asian Cup qualification": 2.0,
    "Gold Cup qualification": 1.5,
    "Friendly": 1.0,
}
DEFAULT_IMPORTANCE = 1.5
WC_BOOST = 8.0                    # weight for 2026 WC matches (vs 1.0 friendlies)

# Elo parameters (eloratings.net conventions)
ELO_K = {  # base K by importance class
    "FIFA World Cup": 60.0,
    "FIFA World Cup qualification": 40.0,
    "UEFA Euro": 50.0,
    "Copa América": 50.0,
    "African Cup of Nations": 50.0,
    "AFC Asian Cup": 50.0,
    "Gold Cup": 40.0,
    "CONCACAF Championship": 40.0,
    "UEFA Nations League": 40.0,
    "Friendly": 20.0,
}
ELO_K_DEFAULT = 30.0
ELO_HOME_ADV = 80.0               # Elo points for non-neutral home side
ELO_START = 1500.0

# Goal-model parameters (some are re-tuned by calibrate())
GOAL_CAP = 4          # soft cap: goals above this count 25%
XG_BLEND = 0.45       # weight on xG when xg.csv provides it
SHRINK_GAMES = 8.0    # pseudo-games of Elo-prior shrinkage for attack/defense
SRS_GD_CAP = 3.0
SRS_HOME_ADV = 0.35   # goals, non-neutral

DEFAULTS = {
    "hfa_goals": 0.32,        # log-mu host advantage (hosts playing in own country)
    "elo_gd_scale": 250.0,    # Elo diff -> expected goal difference divisor
    "elo_blend": 0.45,        # weight of Elo-implied GD vs attack/defense GD
    "goal_level": 0.0,        # global log-mu shift (calibration)
    "form_beta": 0.05,        # log-mu per form z-score
    "exp_beta": 0.03,         # log-mu per experience z-score
    "depth_beta": 0.03,       # log-mu per squad-depth z-score (if data present)
    "rho": -0.12,             # Dixon-Coles low-score correction
    "market_weight": 0.35,    # weight of de-vigged market 1X2 when odds available
}

# Travel penalty (log-mu) for playing a World Cup in North America
TRAVEL_PENALTY = {
    "CONCACAF": 0.0, "CONMEBOL": -0.01, "UEFA": -0.02,
    "CAF": -0.03, "AFC": -0.03, "OFC": -0.03,
}

MAX_GOALS = 10  # exact-score grid 0..MAX_GOALS

# Action Network per-game model anchors (Jun 10, 2026) used for calibration.
# (home, away, venue_country, mu_h, mu_a, p_home, p_draw, p_away, elo_h, elo_a)
AN_ANCHORS = [
    ("Mexico", "South Africa", "Mexico", 1.7, 0.6, 0.63, 0.23, 0.14, 1868, 1545),
    ("Czech Republic", "South Korea", "Mexico", 1.2, 1.1, 0.37, 0.28, 0.34, 1670, 1770),
]

ANCHORS_JSON = DATA / "anchors.json"


def get_anchors() -> list:
    """Calibration anchors; data/anchors.json overrides AN_ANCHORS if present.

    JSON format: list of objects with keys
      home, away, venue, mu_home, mu_away, p_home, p_draw, p_away
    (elo_home / elo_away optional, only used for the sanity-check printout).
    """
    if ANCHORS_JSON.exists():
        import json
        rows = json.loads(ANCHORS_JSON.read_text(encoding="utf-8"))
        return [(r["home"], r["away"], r["venue"], r["mu_home"], r["mu_away"],
                 r["p_home"], r["p_draw"], r["p_away"],
                 r.get("elo_home", 0), r.get("elo_away", 0)) for r in rows]
    return AN_ANCHORS
