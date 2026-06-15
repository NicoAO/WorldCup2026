# World Cup 2026 Prediction System

SRS-family model that predicts every 2026 World Cup match and converts
expected goals into **exact-score probabilities** (Dixon-Coles Poisson).


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
