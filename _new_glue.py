"""Browser glue: thin JSON API over wc_model, called from the page via Pyodide."""
import json
import random
import sys

sys.path.insert(0, "/wc")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from wc_model import config, data, model  # noqa: E402

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


def _last_matches(team, n=10):
    df = data.load_results()
    played, _ = data.split_played_fixtures(df)
    m = played[(played["home_team"] == team) | (played["away_team"] == team)].copy()
    m = m.sort_values("date").tail(n)
    rows = []
    for _, r in m.iterrows():
        is_home = r["home_team"] == team
        opp = r["away_team"] if is_home else r["home_team"]
        gf = int(r["home_score"]) if is_home else int(r["away_score"])
        ga = int(r["away_score"]) if is_home else int(r["home_score"])
        res = "W" if gf > ga else ("D" if gf == ga else "L")
        rows.append({"date": str(r["date"].date()), "opp": opp, "gf": gf,
                     "ga": ga, "tournament": r["tournament"],
                     "result": res, "is_home": bool(is_home)})
    return list(reversed(rows))


def _head_to_head(home, away, n=6):
    """Last n meetings between the two teams, with a W-D-L record read from the
    `home` argument's perspective. Empty record when they've never met."""
    df = data.load_results()
    played, _ = data.split_played_fixtures(df)
    m = played[((played["home_team"] == home) & (played["away_team"] == away))
               | ((played["home_team"] == away) & (played["away_team"] == home))]
    m = m.sort_values("date").tail(n)
    games = []
    wins = draws = losses = 0
    for _, r in m.iterrows():
        is_home = r["home_team"] == home
        gf = int(r["home_score"]) if is_home else int(r["away_score"])
        ga = int(r["away_score"]) if is_home else int(r["home_score"])
        res = "W" if gf > ga else ("D" if gf == ga else "L")
        if res == "W":
            wins += 1
        elif res == "D":
            draws += 1
        else:
            losses += 1
        games.append({"date": str(r["date"].date()),
                      "home": r["home_team"], "away": r["away_team"],
                      "hs": int(r["home_score"]), "as": int(r["away_score"]),
                      "tournament": r["tournament"], "result": res})
    games.reverse()  # most recent first
    return {"games": games, "w": wins, "d": draws, "l": losses, "n": len(m)}


def predict(home, away, venue="", neutral=None):
    r = model.predict_match(STATE, home, away,
                            venue_country=venue or None, neutral=neutral)
    r["matrix"] = [[float(x) for x in row] for row in r["matrix"]]
    r["last_home"] = _last_matches(home)
    r["last_away"] = _last_matches(away)
    r["h2h"] = _head_to_head(home, away)
    return json.dumps(r, default=float)


def fixtures():
    df = data.load_results()
    played, fx = data.split_played_fixtures(df)
    wc = played[(played["tournament"] == "FIFA World Cup")
                & (played["date"] >= config.WC_START)]

    def _txt(v):
        # A missing cell reads back as float NaN; json.dumps would emit the bare
        # literal `NaN`, which is invalid JSON and aborts the whole page update.
        return "" if pd.isna(v) else str(v)

    rows = []
    for _, r in fx.iterrows():
        # Late knockout slots (3rd-place, final) sit in the feed with NA team
        # names until their participants are decided. Skip them entirely.
        if pd.isna(r["home_team"]) or pd.isna(r["away_team"]):
            continue
        rows.append({"date": str(r["date"].date()), "home": r["home_team"],
                     "away": r["away_team"], "city": _txt(r["city"]),
                     "country": _txt(r["country"]), "neutral": bool(r["neutral"]),
                     "score": None})
    for _, r in wc.iterrows():
        if pd.isna(r["home_team"]) or pd.isna(r["away_team"]):
            continue
        rows.append({"date": str(r["date"].date()), "home": r["home_team"],
                     "away": r["away_team"], "city": _txt(r["city"]),
                     "country": _txt(r["country"]), "neutral": bool(r["neutral"]),
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


def wc_progress():
    df = data.load_results()
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"] >= config.WC_START)]
    done = int(wc["home_score"].notna().sum())
    return json.dumps({"done": done, "total": int(len(wc))})


# =====================================================================
# Knockout bracket Monte Carlo
# Bracket structure for the 48-team WC2026 (matches M73-M104, M103 = 3rd-place).
# spec ('p','1A') = group A winner; ('p','2A') = runner-up;
# spec ('t', set('ABCDF')) = qualifying 3rd-placed team from one of those groups
KO_R32_DEF = [
    ("M73", ("p", "2A"), ("p", "2B")),
    ("M74", ("p", "1E"), ("t", "ABCDF")),
    ("M75", ("p", "1F"), ("p", "2C")),
    ("M76", ("p", "1C"), ("p", "2F")),
    ("M77", ("p", "1I"), ("t", "CDFGH")),
    ("M78", ("p", "2E"), ("p", "2I")),
    ("M79", ("p", "1A"), ("t", "CEFHI")),
    ("M80", ("p", "1L"), ("t", "EHIJK")),
    ("M81", ("p", "1D"), ("t", "BEFIJ")),
    ("M82", ("p", "1G"), ("t", "AEHIJ")),
    ("M83", ("p", "2K"), ("p", "2L")),
    ("M84", ("p", "1H"), ("p", "2J")),
    ("M85", ("p", "1B"), ("t", "EFGIJ")),
    ("M86", ("p", "1J"), ("p", "2H")),
    ("M87", ("p", "1K"), ("t", "DEIJL")),
    ("M88", ("p", "2D"), ("p", "2G")),
]
KO_PAIRS = {
    "M89": ("M74", "M77"), "M90": ("M73", "M75"),
    "M91": ("M76", "M78"), "M92": ("M79", "M80"),
    "M93": ("M83", "M84"), "M94": ("M81", "M82"),
    "M95": ("M86", "M88"), "M96": ("M85", "M87"),
    "M97": ("M89", "M90"), "M98": ("M93", "M94"),
    "M99": ("M91", "M92"), "M100": ("M95", "M96"),
    "M101": ("M97", "M98"), "M102": ("M99", "M100"),
    "M104": ("M101", "M102"),
}
THIRD_SLOTS = [
    ("M74", set("ABCDF")), ("M77", set("CDFGH")), ("M79", set("CEFHI")),
    ("M80", set("EHIJK")), ("M81", set("BEFIJ")), ("M82", set("AEHIJ")),
    ("M85", set("EFGIJ")), ("M87", set("DEIJL")),
]
ALL_MIDS = [m for m, _, _ in KO_R32_DEF] + list(KO_PAIRS.keys())


def _spec_label(spec):
    kind, v = spec
    if kind == "p":
        pos = {"1": "winner", "2": "runner-up", "3": "3rd"}[v[0]]
        return f"Group {v[1]} {pos}"
    return f"3rd: {'/'.join(sorted(v))}"


def live_standings():
    """Real (no simulation) group table + bracket cells that are mathematically locked."""
    teams_df = data.load_teams()
    df = data.load_results()
    played, fixtures_df = data.split_played_fixtures(df)
    wc_played = played[(played["tournament"] == "FIFA World Cup")
                       & (played["date"] >= pd.Timestamp(config.WC_START))]
    wc_fixtures = fixtures_df[fixtures_df["tournament"] == "FIFA World Cup"]

    groups = {}
    for t, row in teams_df.iterrows():
        groups.setdefault(row["group"], []).append(t)
    glookup = {t: g for g, ts in groups.items() for t in ts}

    stats = {t: {"pts": 0, "gf": 0, "ga": 0, "gp": 0} for t in teams_df.index}
    for _, g in wc_played.iterrows():
        h = g["home_team"]; a = g["away_team"]
        if glookup.get(h) != glookup.get(a) or glookup.get(h) is None:
            continue
        hs = int(g["home_score"]); as_ = int(g["away_score"])
        stats[h]["gp"] += 1; stats[a]["gp"] += 1
        stats[h]["gf"] += hs; stats[h]["ga"] += as_
        stats[a]["gf"] += as_; stats[a]["ga"] += hs
        if hs > as_:
            stats[h]["pts"] += 3
        elif as_ > hs:
            stats[a]["pts"] += 3
        else:
            stats[h]["pts"] += 1; stats[a]["pts"] += 1

    # Whole group stage finished once every team has played its 3 group games.
    group_stage_over = all(stats[t]["gp"] >= 3 for t in teams_df.index)
    # Teams that reached the knockout stage: they appear in a WC fixture between
    # two different groups (played or still scheduled). Lets us tell an advancing
    # best-third from an eliminated third once the group stage is complete.
    ko_stage = df[(df["tournament"] == "FIFA World Cup")
                  & (df["date"] >= pd.Timestamp(config.WC_START))]
    ko_teams = set()
    for _, g in ko_stage.iterrows():
        gh, ga = glookup.get(g["home_team"]), glookup.get(g["away_team"])
        if gh is not None and ga is not None and gh != ga:
            ko_teams.update((g["home_team"], g["away_team"]))

    out_groups = {}
    locked_pos = {}
    for gid, gteams in groups.items():
        rows = sorted(gteams, key=lambda t: (-stats[t]["pts"],
                                             -(stats[t]["gf"] - stats[t]["ga"]),
                                             -stats[t]["gf"], t))
        all_played_group = all(stats[t]["gp"] >= 3 for t in gteams)

        if all_played_group:
            # Group is decided — rank by the full tiebreakers (pts, GD, GF) so
            # the top two lock even when they finished level on points.
            positions = {rows[0]: "1", rows[1]: "2"}
            # 4th can never be a best third, so it is out. The 3rd-placed team
            # keeps its "3rd hope" only until the whole group stage ends; after
            # that it has either advanced as a best third ("3") or is out.
            for t in rows[3:]:
                positions[t] = "out"
            third = rows[2]
            if not group_stage_over:
                positions[third] = "X"
            else:
                positions[third] = "3" if third in ko_teams else "out"
        else:
            # Group still in progress — only lock what is mathematically certain.
            max_pts = {t: stats[t]["pts"] + 3 * (3 - stats[t]["gp"]) for t in gteams}
            positions = {}
            for t in gteams:
                others = [u for u in gteams if u != t]
                could_match = sum(1 for u in others if max_pts[u] >= stats[t]["pts"])
                must_beat = sum(1 for u in others if stats[u]["pts"] > max_pts[t])
                if could_match == 0:
                    positions[t] = "1"
                elif could_match == 1:
                    positions[t] = "Q"  # top 2, but 1 vs 2 not yet split
                elif must_beat >= 2:
                    positions[t] = "X"  # cannot finish top 2
            first = next((t for t, p in positions.items() if p == "1"), None)
            if first:
                for t in list(positions):
                    if positions[t] == "Q":
                        positions[t] = "2"

        out_groups[gid] = {
            "rows": [{
                "team": t, "pts": stats[t]["pts"], "gp": stats[t]["gp"],
                "gf": stats[t]["gf"], "ga": stats[t]["ga"],
                "gd": stats[t]["gf"] - stats[t]["ga"],
                "lock": positions.get(t),
            } for t in rows],
            "all_played": all_played_group,
        }
        locked_pos[gid] = {
            "1": next((t for t, p in positions.items() if p == "1"), None),
            "2": next((t for t, p in positions.items() if p == "2"), None),
            "3": None,
        }

    def _lookup(spec):
        kind, v = spec
        if kind == "p":
            return locked_pos.get(v[1], {}).get(v[0])
        return None

    # Shootout winners (keyed "TeamA|TeamB", names sorted) written by the page
    # from ESPN, so games level after extra time still advance the right side.
    shootouts = {}
    sp = config.DATA / "shootouts.json"
    try:
        if sp.exists():
            shootouts = json.loads(sp.read_text(encoding="utf-8"))
    except Exception:
        shootouts = {}

    # Every WC match between two different-group teams is a knockout game. Read
    # the actual fixtures (played and scheduled) straight from the data — that is
    # the source of truth for who plays whom, so we never have to guess the
    # third-place allocation ourselves.
    wc_ko = df[(df["tournament"] == "FIFA World Cup")
               & (df["date"] >= pd.Timestamp(config.WC_START))]
    ko_fix = {}      # frozenset({home, away}) -> result record
    ko_by_team = {}  # team -> list of its knockout fixtures
    for _, g in wc_ko.iterrows():
        h = g["home_team"]; a = g["away_team"]
        gh, ga = glookup.get(h), glookup.get(a)
        if gh is None or ga is None or gh == ga:
            continue
        played = pd.notna(g["home_score"])
        rec = {"home": h, "away": a, "date": g["date"], "played": bool(played),
               "hs": None, "as": None, "winner": None, "pens": False}
        if played:
            hs = int(g["home_score"]); as_ = int(g["away_score"])
            rec["hs"], rec["as"] = hs, as_
            if hs > as_:
                rec["winner"] = h
            elif as_ > hs:
                rec["winner"] = a
            else:  # level after extra time — settled on penalties
                rec["pens"] = True
                rec["winner"] = shootouts.get("|".join(sorted((h, a))))
        ko_fix[frozenset((h, a))] = rec
        ko_by_team.setdefault(h, []).append(rec)
        ko_by_team.setdefault(a, []).append(rec)

    # a team's KO games in date order — its Nth game belongs to the Nth KO round
    for recs in ko_by_team.values():
        recs.sort(key=lambda r: r["date"])
    # round index per bracket cell (R32 = 1, R16 = 2, ...)
    round_of = {mid: 1 for mid, _h, _a in KO_R32_DEF}
    for mid, (h_id, a_id) in KO_PAIRS.items():
        round_of[mid] = max(round_of[h_id], round_of[a_id]) + 1

    def _new_cell(home, away, home_label, away_label):
        return {"home": home, "away": away,
                "home_label": home_label, "away_label": away_label,
                "played": False, "hs": None, "as": None,
                "winner": None, "pens": False}

    def _attach(cell, rec):
        cell["played"] = rec["played"]
        if rec["played"]:
            same = rec["home"] == cell["home"]
            cell["hs"] = rec["hs"] if same else rec["as"]
            cell["as"] = rec["as"] if same else rec["hs"]
            cell["winner"] = rec["winner"]
            cell["pens"] = rec["pens"]

    def _fixture_for(team, rnd):
        recs = ko_by_team.get(team, [])
        return recs[rnd - 1] if team and len(recs) >= rnd else None

    def _fill(cell, home_known, away_known, rnd):
        """Fill a cell from the real fixture of whichever participant is known,
        so a round advances even before both feeders (or a shootout) resolve."""
        anchor = home_known or away_known
        rec = _fixture_for(anchor, rnd)
        if not rec:
            return
        other = rec["away"] if rec["home"] == anchor else rec["home"]
        cell["home"], cell["away"] = (anchor, other) if home_known else (other, anchor)
        _attach(cell, rec)

    bracket_cells = {}
    advancing = {}  # mid -> winning team
    # Round of 32: anchor each cell on its fixed group-position team.
    for mid, hspec, aspec in KO_R32_DEF:
        home = _lookup(hspec)
        cell = _new_cell(home, None, _spec_label(hspec), _spec_label(aspec))
        _fill(cell, home, None, 1)
        bracket_cells[mid] = cell
        if cell["winner"]:
            advancing[mid] = cell["winner"]

    # Later rounds (KO_PAIRS is already in dependency order).
    for mid, (h_id, a_id) in KO_PAIRS.items():
        hw, aw = advancing.get(h_id), advancing.get(a_id)
        cell = _new_cell(hw, aw, f"Winner {h_id}", f"Winner {a_id}")
        _fill(cell, hw, aw, round_of[mid])
        bracket_cells[mid] = cell
        if cell["winner"]:
            advancing[mid] = cell["winner"]

    ko_fixtures = []
    for _, r in wc_fixtures.iterrows():
        h = r["home_team"]; a = r["away_team"]
        if (h in teams_df.index and a in teams_df.index
                and (glookup.get(h) != glookup.get(a) or glookup.get(h) is None)):
            ko_fixtures.append({"home": h, "away": a,
                                "date": str(r["date"].date()),
                                "city": r["city"], "country": r["country"]})

    return json.dumps({
        "groups": out_groups,
        "cells": bracket_cells,
        "ko_fixtures": ko_fixtures,
    }, default=float)


def _score_pmf(home, away, max_g=6):
    r = model.predict_match(STATE, home, away, neutral=True, use_market=False)
    mat = np.asarray(r["matrix"])[:max_g + 1, :max_g + 1]
    s = mat.sum()
    if s <= 0:
        return [(0, 0, 1.0)]
    mat = mat / s
    out = []
    for h in range(max_g + 1):
        for a in range(max_g + 1):
            p = float(mat[h, a])
            if p > 1e-9:
                out.append((h, a, p))
    return out


def _sample_score(pmf, rng):
    r = rng.random()
    cum = 0.0
    for h, a, p in pmf:
        cum += p
        if r < cum:
            return h, a
    return pmf[-1][0], pmf[-1][1]


def _sample_winner(pmf, rng, ha, hb):
    h, a = _sample_score(pmf, rng)
    if h > a:
        return ha
    if a > h:
        return hb
    return ha if rng.random() < 0.5 else hb


def _assign_thirds(qualified_groups, rng):
    """Greedy bipartite + backtracking: map 8 R32 third-place slots to group letters."""
    slots = list(THIRD_SLOTS)
    qg = list(qualified_groups)
    # Sort slots by fewest available candidates first (most-constrained-first heuristic)
    slots.sort(key=lambda s: len(s[1] & set(qg)))
    rng.shuffle(qg)
    remaining = list(qg)
    out = {}

    def go(i):
        if i == len(slots):
            return True
        sid, allowed = slots[i]
        for c in [g for g in remaining if g in allowed]:
            out[sid] = c
            remaining.remove(c)
            if go(i + 1):
                return True
            remaining.append(c)
            del out[sid]
        return False

    if not go(0):
        # fallback: assign whatever fits, leave the rest empty
        for sid, allowed in THIRD_SLOTS:
            if sid in out:
                continue
            for g in qg:
                if g in allowed and g not in out.values():
                    out[sid] = g
                    break
    return out


def simulate_bracket(n_sims=400):
    teams_df = data.load_teams()
    df = data.load_results()
    played, fixtures_df = data.split_played_fixtures(df)
    wc_played = played[(played["tournament"] == "FIFA World Cup")
                       & (played["date"] >= pd.Timestamp(config.WC_START))]
    wc_fixtures = fixtures_df[fixtures_df["tournament"] == "FIFA World Cup"]

    groups = {}
    for t, row in teams_df.iterrows():
        groups.setdefault(row["group"], []).append(t)
    glookup = {t: g for g, ts in groups.items() for t in ts}

    def is_group(h, a):
        return glookup.get(h) == glookup.get(a) and glookup.get(h) is not None

    remaining = [(r["home_team"], r["away_team"]) for _, r in wc_fixtures.iterrows()
                 if is_group(r["home_team"], r["away_team"])]

    # cached score pmfs (per matchup), then on-demand for knockouts
    pmfs = {k: _score_pmf(*k) for k in {(h, a) for h, a in remaining}}
    ko_cache = {}

    def kop(h, a):
        if (h, a) not in ko_cache:
            ko_cache[(h, a)] = _score_pmf(h, a)
        return ko_cache[(h, a)]

    base = {t: [0, 0, 0] for t in teams_df.index}  # pts, gf, ga
    for _, g in wc_played.iterrows():
        h = g["home_team"]; a = g["away_team"]
        hs = int(g["home_score"]); as_ = int(g["away_score"])
        if not is_group(h, a):
            continue
        base[h][1] += hs; base[h][2] += as_
        base[a][1] += as_; base[a][2] += hs
        if hs > as_:
            base[h][0] += 3
        elif as_ > hs:
            base[a][0] += 3
        else:
            base[h][0] += 1; base[a][0] += 1

    rng = random.Random(42)
    pos_counts = {t: [0, 0, 0, 0] for t in teams_df.index}
    qual_counts = {t: 0 for t in teams_df.index}
    round_counts = {t: [0] * 6 for t in teams_df.index}
    match_counts = {mid: {} for mid in ALL_MIDS}
    side_counts = {mid: ({}, {}) for mid in ALL_MIDS}  # (home, away)

    for _ in range(n_sims):
        stats = {t: list(s) for t, s in base.items()}
        for h, a in remaining:
            hs, as_ = _sample_score(pmfs[(h, a)], rng)
            stats[h][1] += hs; stats[h][2] += as_
            stats[a][1] += as_; stats[a][2] += hs
            if hs > as_:
                stats[h][0] += 3
            elif as_ > hs:
                stats[a][0] += 3
            else:
                stats[h][0] += 1; stats[a][0] += 1

        first, second, third = {}, {}, {}
        thirds_pool = []
        for gid, gteams in groups.items():
            ranked = sorted(gteams, key=lambda t: (-stats[t][0],
                                                   -(stats[t][1] - stats[t][2]),
                                                   -stats[t][1], rng.random()))
            first[gid], second[gid], third[gid] = ranked[0], ranked[1], ranked[2]
            for i, tt in enumerate(ranked):
                pos_counts[tt][i] += 1
            qual_counts[ranked[0]] += 1
            qual_counts[ranked[1]] += 1
            thirds_pool.append((gid, ranked[2], stats[ranked[2]]))

        thirds_pool.sort(key=lambda x: (-x[2][0],
                                        -(x[2][1] - x[2][2]),
                                        -x[2][1], rng.random()))
        qual_third_groups = [g for g, _, _ in thirds_pool[:8]]
        for _, t, _ in thirds_pool[:8]:
            qual_counts[t] += 1
        ta = _assign_thirds(qual_third_groups, rng)

        def resolve(mid, spec):
            kind, v = spec
            if kind == "p":
                gid = v[1]
                return first[gid] if v[0] == "1" else (second[gid] if v[0] == "2" else third[gid])
            return third.get(ta.get(mid))

        winners = {}
        for mid, hspec, aspec in KO_R32_DEF:
            ht = resolve(mid, hspec); at = resolve(mid, aspec)
            if ht is None or at is None:
                continue
            match_counts[mid][(ht, at)] = match_counts[mid].get((ht, at), 0) + 1
            side_counts[mid][0][ht] = side_counts[mid][0].get(ht, 0) + 1
            side_counts[mid][1][at] = side_counts[mid][1].get(at, 0) + 1
            round_counts[ht][0] += 1
            round_counts[at][0] += 1
            winners[mid] = _sample_winner(kop(ht, at), rng, ht, at)

        rounds_def = [(1, ["M89", "M90", "M91", "M92", "M93", "M94", "M95", "M96"]),
                      (2, ["M97", "M98", "M99", "M100"]),
                      (3, ["M101", "M102"]),
                      (4, ["M104"])]
        for ridx, mids in rounds_def:
            for mid in mids:
                h_id, a_id = KO_PAIRS[mid]
                ht = winners.get(h_id); at = winners.get(a_id)
                if ht is None or at is None:
                    continue
                match_counts[mid][(ht, at)] = match_counts[mid].get((ht, at), 0) + 1
                side_counts[mid][0][ht] = side_counts[mid][0].get(ht, 0) + 1
                side_counts[mid][1][at] = side_counts[mid][1].get(at, 0) + 1
                round_counts[ht][ridx] += 1
                round_counts[at][ridx] += 1
                winners[mid] = _sample_winner(kop(ht, at), rng, ht, at)
        champ = winners.get("M104")
        if champ is not None:
            round_counts[champ][5] += 1

    out = {"n_sims": n_sims, "remaining_group_matches": len(remaining),
           "groups": {}, "matches": {}, "teams": {}}
    for gid, gteams in groups.items():
        rows = []
        for t in gteams:
            pc = pos_counts[t]
            rows.append({"team": t,
                         "p1": pc[0] / n_sims, "p2": pc[1] / n_sims,
                         "p3": pc[2] / n_sims, "p4": pc[3] / n_sims,
                         "qualify": qual_counts[t] / n_sims,
                         "locked_in": qual_counts[t] == n_sims,
                         "eliminated": qual_counts[t] == 0})
        rows.sort(key=lambda x: -(x["p1"] + x["p2"]))
        out["groups"][gid] = rows

    for mid in ALL_MIDS:
        pairs = sorted(match_counts[mid].items(), key=lambda x: -x[1])[:5]
        home_top = sorted(side_counts[mid][0].items(), key=lambda x: -x[1])[:3]
        away_top = sorted(side_counts[mid][1].items(), key=lambda x: -x[1])[:3]
        out["matches"][mid] = {
            "pairs": [{"home": h, "away": a, "prob": c / n_sims} for (h, a), c in pairs],
            "home": [{"team": t, "prob": c / n_sims} for t, c in home_top],
            "away": [{"team": t, "prob": c / n_sims} for t, c in away_top],
        }

    for t, rc in round_counts.items():
        out["teams"][t] = {"r32": rc[0] / n_sims, "r16": rc[1] / n_sims,
                           "qf": rc[2] / n_sims, "sf": rc[3] / n_sims,
                           "final": rc[4] / n_sims, "champion": rc[5] / n_sims}
    return json.dumps(out, default=float)
