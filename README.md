# World Cup 2026 Prediction System

Advanced SRS-family model that predicts every 2026 World Cup match and converts
expected goals into **exact-score probabilities** (Dixon-Coles Poisson).

Inspired by the structure of [Action Network's matchup tool](https://www.actionnetwork.com/soccer/2026-fifa-world-cup-matchup-tool-predictions-odds-trends-any-match)
(win probability + projected score + goal total, built on FIFA ranks, Elo,
head-to-head, form, and sportsbook lines) and calibrated against their
published day-1 projections.

## Website (GitHub Pages) — recommended

The whole model runs **in the browser** (Pyodide/WebAssembly) — no server needed.

**Deploy once:**
1. Create a new GitHub repository (public), e.g. `wc2026-predictor`.
2. Upload the **contents** of this `worldcup2026` folder to the repo root
   (`index.html` must sit at the top level; keep `.nojekyll` — without it
   GitHub drops `wc_model/__init__.py` and the site breaks).
3. Repo → Settings → Pages → "Deploy from a branch" → branch `main`, folder `/ (root)` → Save.
4. Open `https://<your-username>.github.io/wc2026-predictor/` (first build ~1 min).

**Use daily:**
- **🔄 Update results & odds** — pulls the latest scores (new WC results appear
  in the community dataset within hours of full time) + fresh sportsbook lines,
  retrains in ~20s. New WC matches weigh 8× friendlies.
- **Matches tab** — click any fixture for win/draw/loss, projected score,
  most-likely scorelines and the exact-score matrix.
- **Matchup tool** — compare any two of the 48 teams, neutral or host venue.
- **Settings** — paste your The Odds API key once (stays in your browser's
  localStorage, never in the public repo) and manage calibration anchors.

## Quick start (local CLI)

```powershell
python train.py                      # fit ratings + calibrate (run once, ~30s)
python predict.py --day 2026-06-11   # full slate for a date
python predict.py "Brazil" "Morocco" --neutral        # any matchup
python predict.py "Mexico" "South Korea" --venue Mexico  # host edge applies
python update.py                     # pull new WC scores, retrain (run daily)
```

## What goes into a prediction

| Requirement | Where it lives |
|---|---|
| FIFA ranking/points | `data/teams.csv` (official 11 Jun 2026 release) → composite strength |
| Last 10 games / form | `features.form_last10` — ppg, W/D/L string, Elo-adjusted over/under-performance → small xG adjustment |
| Goal-difference performance | weighted **SRS** on capped goal difference (`ratings.compute_srs`) |
| Team strength / strength difference | internal **Elo** (eloratings.net rules, validated vs Action Network's published Elo) blended into the goal-difference projection |
| Attack vs opponent defense | weighted Poisson fit `log μ = c + att_team − def_opp + …` (`ratings.fit_attack_defense`) |
| Qualification strength | ppg/GD in 2026 qualifying (`features.qualification_strength`), and qualifiers weigh 3× friendlies in training |
| Host & travel context | calibrated host bump (hosts playing in-country, +0.13 log-goals ≈ Elo +80) and confederation travel penalties |
| Squad experience | weighted major-tournament minutes proxy since 2014 |
| Squad depth | optional `data/squad_values.csv` (team,value_meur — e.g. from Transfermarkt); neutral if absent |
| xG | actual goals are soft-capped (blowout damping, an xG-style smoothing); drop per-match xG into `data/xg.csv` (date,team,opponent,xg) to blend real xG at 45% |
| Betting odds | `wc_model/odds.py` — The Odds API + manual CSV (below) |
| Exact-score probabilities | Dixon-Coles-corrected Poisson grid (`wc_model/scoreline.py`) |

## World Cup results weigh more

`update.py` re-downloads the results dataset (martj42/international_results,
which fills in WC scores as games finish) and retrains. 2026 WC matches enter
with importance **8.0 vs 1.0 for friendlies** (qualifiers 3.0, Nations League
2.0) on top of a 2-year-half-life time decay — so tournament results quickly
dominate form and ratings. Run `python update.py` each morning.

## Betting odds integration

1. Free key from https://the-odds-api.com (500 req/month), then
   `$env:ODDS_API_KEY = "yourkey"` — predictions automatically fetch h2h lines
   (median across books, de-vigged) and blend them at 35% weight
   (`market_weight` in `wc_model/config.py`).
2. No key? Copy `data/manual_odds.sample.csv` to `data/manual_odds.csv` and
   type in real sportsbook decimal odds per match.

The exact-score matrix is re-scaled so its win/draw/loss mass matches the
blended probabilities.

## Calibration anchors — what to paste

The model's global parameters (`goal_level`, `hfa_goals`, `elo_gd_scale`,
`elo_blend`) are tuned so the model reproduces Action Network's published
per-game projections. Each anchor is one match where they published
**projected score** and **win/draw/loss probabilities** (their daily
"Matches and Predictions" articles). Format (website Settings tab, or
`data/anchors.json`, or `AN_ANCHORS` in `wc_model/config.py`):

```json
{ "home": "Mexico", "away": "South Africa", "venue": "Mexico",
  "mu_home": 1.7, "mu_away": 0.6,
  "p_home": 0.63, "p_draw": 0.23, "p_away": 0.14 }
```

- `mu_home`/`mu_away` = their projected score, e.g. "Projected score: Mexico 1.7, South Africa 0.6"
- `p_*` = their win/draw/loss percentages as decimals
- `venue` = country the match is played in (host edge applies when a host plays at home)

More anchors = a better-calibrated model. After pasting, hit **Recalibrate**
on the website (or `python train.py` locally).

## Files

```
wc_model/config.py     all knobs (weights, decay, importance, calibration anchors)
wc_model/data.py       load results, weights, goal capping, xG blend
wc_model/ratings.py    Elo, weighted SRS, attack/defense Poisson
wc_model/features.py   form, qualification, experience, squad depth
wc_model/scoreline.py  exact-score matrix, 1X2/totals/BTTS, market blending
wc_model/odds.py       The Odds API client + manual odds
wc_model/model.py      training, calibration, prediction orchestration
data/results.csv       49k+ internationals since 1872 incl. WC2026 fixtures
data/teams.csv         48 teams: group, FIFA rank/points, confederation, host
```

Caveats: international xG isn't freely available, so the model uses smoothed
actual goals unless you supply `data/xg.csv`; squad depth is neutral until you
add market values; knockout predictions are 90-minute probabilities (draws
possible — extra time/penalties not modeled).
