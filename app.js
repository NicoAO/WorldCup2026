/* WC2026 Predictor — runs the Python model in-browser via Pyodide. */
"use strict";

const RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv";
const ODDS_SPORTS = ["soccer_fifa_world_cup", "soccer_fifa_world_cup_2026"];
const PY_FILES = ["__init__.py", "config.py", "data.py", "ratings.py",
                  "features.py", "scoreline.py", "odds.py", "model.py"];
const DEFAULT_ANCHORS = [
  { home: "Mexico", away: "South Africa", venue: "Mexico",
    mu_home: 1.7, mu_away: 0.6, p_home: 0.63, p_draw: 0.23, p_away: 0.14 },
  { home: "Czech Republic", away: "South Korea", venue: "Mexico",
    mu_home: 1.2, mu_away: 1.1, p_home: 0.37, p_draw: 0.28, p_away: 0.34 },
];

let py = null;          // pyodide instance
let fixtures = [];      // [{date,home,away,city,country,neutral,score}]
let teams = [];         // ratings table rows

const $ = (id) => document.getElementById(id);
const status = (msg) => { $("status").textContent = msg; };
const progress = (p) => { $("progressfill").style.width = `${p}%`; };

/* ---------------------------------------------------------- boot */
async function boot() {
  try {
    progress(5); status("loading Python runtime (first load ~20s)…");
    py = await loadPyodide();
    progress(25); status("loading numpy / pandas / scipy…");
    await py.loadPackage(["numpy", "pandas", "scipy"]);
    progress(45); status("fetching model files…");

    py.FS.mkdirTree("/wc/wc_model");
    py.FS.mkdirTree("/wc/data");
    for (const f of PY_FILES) {
      const txt = await fetchText(`wc_model/${f}?v=${Date.now()}`);
      py.FS.writeFile(`/wc/wc_model/${f}`, txt);
    }
    py.FS.writeFile("/wc/data/teams.csv", await fetchText(`data/teams.csv?v=${Date.now()}`));

    // restore persisted settings (fall back to the calibration shipped in the repo)
    let cal = localStorage.getItem("calibration");
    if (!cal) cal = await fetchText("data/calibration.json").catch(() => null);
    if (cal) py.FS.writeFile("/wc/data/calibration.json", cal);
    const anchors = localStorage.getItem("anchors") ||
                    JSON.stringify(DEFAULT_ANCHORS, null, 2);
    py.FS.writeFile("/wc/data/anchors.json", anchors);
    $("anchors-box").value = anchors;
    $("odds-key").value = localStorage.getItem("oddsKey") || "";

    py.runPython(await fetchText(`web/glue.py?v=${Date.now()}`));
    progress(55);
    await updateAll(true);

    $("btn-update").disabled = false;
  } catch (err) {
    status(`error: ${err.message ?? err}`);
    console.error(err);
  }
}

async function fetchText(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`fetch failed: ${url} (${r.status})`);
  return await r.text();
}

/* ------------------------------------------- update results + odds */
async function updateAll(firstLoad = false) {
  $("btn-update").disabled = true;
  try {
    status("downloading latest results…");
    let csv;
    try {
      csv = await fetchText(`${RESULTS_URL}?t=${Date.now()}`);
      if (!csv.startsWith("date,home_team")) throw new Error("bad csv");
    } catch {
      status("live download failed — using bundled results.csv");
      csv = await fetchText("data/results.csv");
    }
    py.FS.writeFile("/wc/data/results.csv", csv);
    progress(65);

    await fetchOdds();
    progress(75);

    // retrain unless we have a cached state for this exact dataset
    const cacheKey = `${csv.length}|${localStorage.getItem("calibration") || "def"}`;
    const cached = localStorage.getItem("ratingsCache");
    if (!firstLoad || !cached || JSON.parse(cached).key !== cacheKey) {
      status("training model in browser (Elo + SRS + Poisson, ~20s)…");
      await new Promise((res) => setTimeout(res, 30)); // let UI paint
      py.runPython("retrain()");
      try {
        const ratings = py.FS.readFile("/wc/data/ratings.json", { encoding: "utf8" });
        localStorage.setItem("ratingsCache", JSON.stringify({ key: cacheKey, ratings }));
      } catch { /* localStorage may be full — fine */ }
    } else {
      status("restoring cached ratings…");
      py.FS.writeFile("/wc/data/ratings.json", JSON.parse(cached).ratings);
      py.runPython("load_cached_state()");
    }
    progress(90);

    fixtures = JSON.parse(py.runPython("fixtures()"));
    teams = JSON.parse(py.runPython("teams_table()"));
    const prog = JSON.parse(py.runPython("wc_progress()"));
    $("wc-progress").textContent =
      `${prog.done}/${prog.total} WC matches played — completed games weigh 8× friendlies in training`;

    renderDaySelect();
    renderFixtures();
    renderRatings();
    renderMatchupSelectors();
    progress(100);
    status(`ready — ratings as of ${py.runPython("STATE['asof']")}`);
  } catch (err) {
    status(`update error: ${err.message ?? err}`);
    console.error(err);
  } finally {
    $("btn-update").disabled = false;
  }
}

async function fetchOdds() {
  const key = (localStorage.getItem("oddsKey") || "").trim();
  if (!key) { $("odds-status").textContent = "no API key saved — model runs without market blend"; return 0; }
  for (const sport of ODDS_SPORTS) {
    try {
      status("fetching sportsbook odds…");
      const url = `https://api.the-odds-api.com/v4/sports/${sport}/odds?regions=eu,us&markets=h2h&oddsFormat=decimal&apiKey=${key}`;
      const r = await fetch(url);
      if (!r.ok) continue;
      const events = await r.json();
      if (!events.length) continue;
      const remaining = r.headers.get("x-requests-remaining");
      py.FS.writeFile("/wc/data/odds_cache.json",
        JSON.stringify({ fetched_at: Date.now() / 1000, events }));
      $("odds-status").textContent =
        `✓ odds loaded: ${events.length} matches priced (${remaining} API requests left this month)`;
      return events.length;
    } catch (e) { console.warn("odds fetch failed", e); }
  }
  $("odds-status").textContent = "odds fetch failed — check key / quota";
  return 0;
}

/* ------------------------------------------------------ rendering */
function renderDaySelect() {
  const days = [...new Set(fixtures.map((f) => f.date))].sort();
  const today = new Date().toISOString().slice(0, 10);
  const sel = $("day-select");
  sel.innerHTML = days.map((d) => `<option value="${d}">${d}</option>`).join("");
  sel.value = days.includes(today) ? today
            : days.find((d) => d >= today) || days[days.length - 1];
}

function renderFixtures() {
  const day = $("day-select").value;
  const list = fixtures.filter((f) => f.date === day);
  $("fixture-list").innerHTML = list.map((f, i) => `
    <div class="fixture" data-i="${fixtures.indexOf(f)}">
      <span class="teams">${f.home} vs ${f.away}
        ${f.score ? `<span class="score"> ${f.score} FT</span>` : ""}</span>
      <span class="where">${f.city}, ${f.country}${f.neutral ? "" : " · host"}</span>
    </div>`).join("") || `<p class="muted">No fixtures on ${day}.</p>`;
  $("match-prediction").innerHTML = "";
  document.querySelectorAll(".fixture").forEach((el) =>
    el.addEventListener("click", () => predictFixture(+el.dataset.i)));
}

async function predictFixture(i) {
  const f = fixtures[i];
  await renderPrediction("match-prediction", f.home, f.away, f.country, f.neutral,
    f.score ? `Final score was ${f.score} — prediction shown as the model saw it pre-match.` : "");
}

async function renderPrediction(targetId, home, away, venue, neutral, note) {
  status(`predicting ${home} vs ${away}…`);
  const fn = py.globals.get("predict");
  const r = JSON.parse(fn(home, away, venue || "", neutral ?? null));
  fn.destroy?.();
  status(`ready — ratings as of ${py.runPython("STATE['asof']")}`);

  const [mh, md, ma] = r.final ? [r.final.p_home, r.final.p_draw, r.final.p_away] : r.model_1x2;
  const pct = (x) => `${Math.round(100 * x)}%`;
  const market = r.market_1x2
    ? `<div class="kv">Market (no-vig): <b>${pct(r.market_1x2[0])} / ${pct(r.market_1x2[1])} / ${pct(r.market_1x2[2])}</b>
       — blended into final at 35%</div>`
    : `<div class="kv muted">no sportsbook odds for this matchup (save an API key in Settings)</div>`;
  const scorebars = r.top_scores.map(([s, p]) => `
    <div class="scorebar"><span class="lbl">${s}</span>
      <span class="pct">${(100 * p).toFixed(1)}%</span>
      <div class="fill" style="width:${Math.min(100, 3.2 * 100 * p)}px"></div></div>`).join("");

  const n = 6;
  let matrix = `<table class="matrix"><tr><th>${home} ↓ \\ ${away} →</th>`;
  for (let j = 0; j < n; j++) matrix += `<th>${j}</th>`;
  matrix += "</tr>";
  const maxP = Math.max(...r.matrix.slice(0, n).flatMap((row) => row.slice(0, n)));
  for (let i2 = 0; i2 < n; i2++) {
    matrix += `<tr><th>${i2}</th>`;
    for (let j = 0; j < n; j++) {
      const p = r.matrix[i2][j];
      const alpha = (p / maxP) * 0.85;
      matrix += `<td style="background:rgba(56,178,106,${alpha.toFixed(3)})">${(100 * p).toFixed(1)}</td>`;
    }
    matrix += "</tr>";
  }
  matrix += "</table>";

  const fh = r.form[0], fa = r.form[1];
  $(targetId).innerHTML = `
    <div class="card">
      <h2>${home} vs ${away}</h2>
      ${note ? `<div class="kv" style="color:var(--accent2)">${note}</div>` : ""}
      <div class="kv">Elo <b>${Math.round(r.elo[0])}</b> vs <b>${Math.round(r.elo[1])}</b>
        · strength <b>${(50 + 15 * r.strength[0]).toFixed(0)}</b> vs <b>${(50 + 15 * r.strength[1]).toFixed(0)}</b></div>
      ${fh && fa ? `<div class="kv">Form: <b>${fh.wdl}</b> (${fh.ppg.toFixed(2)} ppg) vs <b>${fa.wdl}</b> (${fa.ppg.toFixed(2)} ppg)</div>` : ""}
      <div class="scoreproj">${home} <span class="mu">${r.mu_home.toFixed(2)}</span>
        – <span class="mu">${r.mu_away.toFixed(2)}</span> ${away}</div>
      <div class="bar3">
        <div class="h" style="width:${100 * mh}%">${home} ${pct(mh)}</div>
        <div class="d" style="width:${100 * md}%">draw ${pct(md)}</div>
        <div class="a" style="width:${100 * ma}%">${away} ${pct(ma)}</div>
      </div>
      ${market}
      <div class="kv">Over 2.5: <b>${pct(r.final.over_2_5)}</b> · BTTS: <b>${pct(r.final.btts)}</b></div>
      <div class="scorebars"><b>Most likely scorelines</b>${scorebars}</div>
      <details><summary class="muted">exact-score matrix (%)</summary>${matrix}</details>
    </div>`;
}

function renderRatings() {
  $("ratings-table").querySelector("tbody").innerHTML = teams.map((t, i) => `
    <tr><td>${i + 1}</td><td>${t.team}</td><td>${t.group}</td>
    <td><b>${t.strength}</b></td><td>${t.elo}</td><td>#${t.fifa_rank}</td>
    <td style="font-family:monospace">${t.form}</td><td>${t.ppg}</td></tr>`).join("");
}

function renderMatchupSelectors() {
  const names = [...teams].sort((a, b) => a.team.localeCompare(b.team));
  const opts = names.map((t) => `<option>${t.team}</option>`).join("");
  $("team-a").innerHTML = opts;
  $("team-b").innerHTML = opts;
  $("team-a").value = "Brazil";
  $("team-b").value = "Argentina";
}

/* ------------------------------------------------------- settings */
async function saveKey() {
  localStorage.setItem("oddsKey", $("odds-key").value.trim());
  await fetchOdds(); // odds cache is read at predict time — no retrain needed
}

async function recalibrate() {
  $("cal-status").textContent = "recalibrating…";
  try {
    const anchors = JSON.parse($("anchors-box").value); // validate
    const txt = JSON.stringify(anchors, null, 2);
    localStorage.setItem("anchors", txt);
    py.FS.writeFile("/wc/data/anchors.json", txt);
    await new Promise((res) => setTimeout(res, 30));
    const out = JSON.parse(py.runPython("recalibrate()"));
    const cal = py.FS.readFile("/wc/data/calibration.json", { encoding: "utf8" });
    localStorage.setItem("calibration", cal);
    localStorage.removeItem("ratingsCache");
    $("cal-status").textContent = "✓ recalibrated";
    $("cal-result").textContent =
      "tuned parameters:\n" + JSON.stringify(out.params, null, 2) +
      "\n\nanchor fit:\n" + out.checks.map((c) =>
        `${c.match}: model ${c.model} (target ${c.target}) | ` +
        `1X2 ${c.p.map((x) => Math.round(100 * x) + "%").join("/")} ` +
        `(target ${c.p_target.map((x) => Math.round(100 * x) + "%").join("/")})`).join("\n");
    teams = JSON.parse(py.runPython("teams_table()"));
    renderRatings();
  } catch (err) {
    $("cal-status").textContent = `error: ${err.message ?? err}`;
  }
}

/* ---------------------------------------------------------- wiring */
document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
  document.querySelectorAll(".tabpane").forEach((x) => x.classList.remove("active"));
  b.classList.add("active");
  $(`tab-${b.dataset.tab}`).classList.add("active");
}));
$("btn-update").addEventListener("click", () => updateAll());
$("day-select").addEventListener("change", renderFixtures);
$("btn-matchup").addEventListener("click", () => renderPrediction(
  "matchup-prediction", $("team-a").value, $("team-b").value,
  $("venue-select").value, $("venue-select").value ? null : true, ""));
$("btn-save-key").addEventListener("click", saveKey);
$("btn-recalibrate").addEventListener("click", recalibrate);

boot();
