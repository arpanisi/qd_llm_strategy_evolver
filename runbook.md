# Runbook — qd_llm_strategy_evolver

Implementation runbook: how to set up the environment, build the data artifacts,
and run the three evolution tiers. This is the operational companion to the
abstract-style README; it assumes you are at the repo root
(`evolve-agent/qd_llm_strategy_evolver`) and know the design from the coding plan.

All commands use `.venv/bin/python`. The project targets Python 3.12 (the lock
files and `zipline-reloaded` support 3.9–3.12; this repo was developed on 3.12).

---

## 1. Environment setup

### 1.1 Python + dependencies

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` pins the major packages:

- `zipline-reloaded>=3.0` — backtest engine
- `wrds`, `yfinance` — data sources (equities / futures)
- `pandas`, `numpy`, `scipy`, `pyarrow` — data + numerics
- `PyYAML`, `python-dotenv` — config + env loading
- `openai>=1.30` — OpenRouter chat client
- `pytest>=8.0` — tests

### 1.2 `.env` credentials

The code reads `KEY=VALUE` lines from a `.env` file at the repo root
(`src/config/env.py`). Create it with the four locked credential names:

```
OPENROUTER_API_KEY=<openrouter key>   # required for any evolution run
DEEPSEEK_API_KEY=<deepseek key>       # optional/reserved
WRDS_USERID=<wrds username>           # required only to re-fetch equities data
WRDS_PGPASS=<wrds password>           # required only to re-fetch equities data
```

Only `OPENROUTER_API_KEY` is needed to run the evolution tiers. The WRDS
credentials are only used by `scripts/fetch_data.py --track equities` to
re-pull the CRSP universe.

### 1.3 Config

`config/evolver.yaml` holds every runtime parameter: tracks, evolution, path A/B
parameters, and model IDs. The tier specs that override `evolution.generations`,
`candidates_per_island_per_gen`, `migration_interval`, and `curation_interval`
live in `tier_specs()` in `src/evolution/orchestrator.py`:

| tier | G | candidates/island/gen | migration interval | curation interval |
|------|---|----------------------|--------------------|-------------------|
| tier1 (smoke) | 1 | 1 | — | — |
| tier2 (small) | 8 | 2 | 4 | 8 |
| tier3 (full) | 150 | 8 | 10 | 50 |

`cfg.evolution.num_islands` (9) is always taken from config and is not tier
overridden.

---

## 2. Data setup (Step 1)

Two tracks run independently: **equities** (CRSP 15-name universe, daily bars,
XNYS) and **futures** (ES + NQ front-month continuous, daily bars, CME).

### 2.1 Fetch raw data

```bash
.venv/bin/python scripts/fetch_data.py --track all
```

Writes `data/raw/crsp_equities.parquet` and `data/raw/futures.parquet`. Equities
requires WRDS credentials; futures uses `yfinance`. Re-fetch with `--force`.

### 2.2 Ingest Zipline bundles (one-time)

Each track's data must be ingested into Zipline's bcolz bundle format before any
backtest runs. Bundles are already ingested in this environment
(`~/.zipline/data/{equities,futures}`). On a fresh machine:

```bash
.venv/bin/python -c "
from src.config import load_config
from src.engine.bundle import ensure_bundles_ingested
ensure_bundles_ingested(load_config())
"
```

---

## 3. Build the bootstrap artifacts

These must run in order — each consumes the previous stage's output. The
`outputs/` directories are created as needed.

### 3.1 Baselines (Step 11)

```bash
.venv/bin/python scripts/run_baselines.py
```

Writes `outputs/baselines/<track>/baselines.csv` and the equal-weighted
benchmark return series `rb_returns_{train,validation,test}.csv`. The
benchmark series are the Information-Ratio benchmark `R_b` used by every
candidate backtest and by the final selection reruns.

### 3.2 Seed strategies (Step 1)

```bash
.venv/bin/python scripts/run_seeds.py
```

Backtests all 9 seed strategies per track over the training window and writes
`outputs/seeds/<track>/seeds.csv`. The seed metrics fix Path A's feature-map
bin edges (Step 2A) and seed Path B's ideal point (Step 2B).

### 3.3 Search setup (Steps 2A / 2B)

```bash
.venv/bin/python scripts/setup_search.py
```

For each track: freezes the 5 continuous feature-map bin edges (seed range
expanded by `path_a.bin_headroom`, cut into `path_a.bins_per_dim` bins — never
recomputed later), places the 9 seed entries into the archive, generates the
Das–Dennis reference directions (`p=5`, `M=4` → 56 points), seeds the running
ideal point, and writes the population history. Outputs to
`outputs/search/<track>/` (`bin_edges.json`, `archive_log.csv`,
`archive_cells.csv`, `reference_points.csv`, `ideal_point.json`, ...).

Evolution runs load these frozen artifacts (`src/evolution/persistence.py`), so
an evolution run continues exactly where setup left off.

---

## 4. Evolution tiers (Steps 3–10)

Tiers run both paths (A: feature-map archive, B: reference-point population) on
both tracks. The two tracks and the two paths run sequentially; **the 9 islands
within a (track, path) runner run concurrently** across worker processes
(see §4.2).

```bash
.venv/bin/python scripts/run_tier1.py     # smoke: G=1, 1 candidate/island
.venv/bin/python scripts/run_tier2.py     # small:  G=8, 2 candidates/island
.venv/bin/python scripts/run_tier3.py     # full:   G=150, 8 candidates/island
```

Per runner, the engine writes to `outputs/evolution/<track>/<path>/<tier>/`:
archive/population CSVs per generation, `records.csv` (Path A) or
`population_history.csv` / `final_population.csv` (Path B),
`migration_events.csv`, `run.json`, and a final `summary.json` from `finish()`.

### 4.1 The candidate pipeline (Steps 5/6/7)

Every candidate on an island flows through:

1. **Step 5 research** — the research role (deepseek) proposes a hypothesis
   JSON conditioned on the sampled parent + 7 cousins + the island's insight
   log.
2. **Step 6 implementation** — the coder role writes Zipline code, which is
   backtested over the training window under the real cost model; on a
   crash / syntax error / zero-trades the code is refined (cap:
   `models.max_refinement_attempts`).
3. **Step 7 evaluation** — the evaluator role scores the backtest and returns
   style categories + an insight.
4. **Step 8A/8B placement** — Path A places into the feature map (near-tie
   broken by turnover); Path B feeds the island's incoming pool for
   environmental selection.

Sampling and placement happen on the single main thread; Steps 5/6/7 run in
worker processes (see §4.2).

### 4.2 Concurrency model

Per the locked Runtime Plan, the 9 islands run concurrently, while candidate
generation stays strictly sequential *within* an island. Implementation:

- `src/evolution/worker.py` exposes `produce_candidate(task)` — a pure function
  that runs Steps 5/6/7 from a self-contained, picklable `task` dict
  (sampled parent/cousins, insight lines, config, benchmark series). It never
  touches feature-map / population / reference-table state.
- `BaseRunner.run()` (`src/evolution/orchestrator.py`) samples parents/cousins
  on the main thread, dispatches one worker per island per candidate slot to a
  persistent `ProcessPoolExecutor` (one worker per island, up to
  `os.cpu_count()`), then collects results in island order and does **all**
  archive/population writes on the main thread — so no locking is needed and no
  archive-cell / reference-point race is possible.
- Migration and insight curation also run on the main thread at their intervals.

### 4.3 Models

All LLM calls go to OpenRouter (`models.openrouter_base_url`) — **no local
GPU or model inference is required**. The local workload is Zipline backtests
(CPU-bound; 9 worker processes run backtests concurrently).

Current model configuration (see the deviation note in `config/evolver.yaml`):

| role | model |
|------|-------|
| research (Step 5) | `deepseek/deepseek-v3.2` |
| implementation (Step 6) | `deepseek/deepseek-v3.2` |
| evaluation (Step 7) | `deepseek/deepseek-v3.2` |

> **Recorded deviation:** the locked plan specifies a cheap ~30B-class
> open-weight instruct model for the high-volume implementation role to keep
> Step 6 cheap at Tier 3 scale (43,200 candidates). Both ~30B-class candidates
> tested on OpenRouter failed the real coder prompts: `qwen3.6-35b-a3b` (MoE,
> ~3B active) returned empty completions at a high rate, and the dense
> `qwen-2.5-coder-32b-instruct` deterministically truncated to 31 chars on
> long prompts (parent + cousin history). `deepseek-v3.2` is reliable on the
> same prompts. This trades Step 6 cost for reliability; it is documented
> explicitly in `config/evolver.yaml` and should be revisited before a
> cost-sensitive full run.

---

## 5. Step 12 comparison report

```bash
.venv/bin/python scripts/run_comparison.py tier2     # default tier2
.venv/bin/python scripts/run_comparison.py tier3
```

Per track, loads both paths' `summary.json`, computes average test-window
turnover, and runs the decisive dominance check (does any Path B Pareto member
dominate a Path A final on Sharpe/Sortino/Total Return/Max Drawdown?). Writes
`outputs/reports/<track>/path_comparison_<tier>.json` and prints a summary.

---

## 5.1 Step 13 backtest-validation (PBO / DSR) — Tier 3 deliverable

```bash
.venv/bin/python scripts/run_validation.py tier3     # default tier3
```

Per track, computes two multiple-testing-corrected significance diagnostics
entirely from persisted data (no re-backtest):

- **PBO** (Probability of Backtest Overfitting) via CSCV with `S=16`
  contiguous blocks (`C(16,8)=12,870` block splits), both paths. Scope: Path A
  = every occupied feature-map cell's entry; Path B = the final
  validation-confirmed Pareto set. Requires each scope strategy's
  training-window daily returns.
- **DSR** (Deflated Sharpe Ratio), Path A only: the per-island-best strategy
  (already test-rerun by Step 8A) with the highest validation-window Combined
  Score, using its test-window Sharpe and daily returns, the raw Archive Log
  trial count as `N`, and IR's cross-trial std as `σ_SR_trials`.

Writes `outputs/reports/<track>/backtest_validation_<tier>.json` with every
intermediate (SR_hat, SR_0, σ_hat, N, σ_trials) plus the final numbers.
Degenerate cases (e.g. `N < 2`, zero cross-trial IR spread) report
`undefined` with a reason.

**Prerequisites:** the daily returns are persisted *during* the run — seed
returns by `scripts/run_seeds.py` (`outputs/search/<track>/returns/`), and
every candidate's train-window returns by the tier run
(`outputs/evolution/<track>/<path>/<tier>/returns/`). The Path A test-window
returns for the DSR target are written at `finish()` time
(`outputs/evolution/<track>/a/<tier>/test_returns/`). Runs started before this
persistence existed (e.g. the original Tier 1 artifacts) report `undefined`.
Step 13 is only required at Tier 3 per the Runtime Plan.

---

## 6. Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

Covers Path A (bin freezing, placement, sampling, breadth), Path B
(Das–Dennis refs, normalization, non-dominated sorting, environmental
selection, breadth), Step 10 average one-way turnover, Step 13 (PBO via CSCV,
DSR, and the end-to-end validation report), the coder refinement
loop, the research/evaluation role JSON handling, the mating/report helper,
and the concurrency refactor (worker purity contract, the shared Steps 5/6/7
pipeline, and the multi-generation concurrent `run()` with migration +
curation).

---

## 7. Outputs layout

```
outputs/
  baselines/<track>/rb_returns_{train,validation,test}.csv   # IR benchmark R_b
  seeds/<track>/seeds.csv                                    # seed metrics (Step 1)
  search/<track>/returns/<sid>.csv                           # seed train returns (Step 13)
  search/<track>/                                            # frozen setup (Steps 2A/2B)
  evolution/<track>/<path>/<tier>/returns/<sid>.csv          # candidate train returns (Step 13)
  evolution/<track>/<path>/<tier>/test_returns/<sid>.csv     # Path A test returns (Step 13 DSR)
  evolution/<track>/<path>/<tier>/                           # per-run artifacts
  reports/<track>/path_comparison_<tier>.json                # Step 12 report
  reports/<track>/backtest_validation_<tier>.json            # Step 13 report
```

Source strategy code for each generated candidate is also written to
`src/strategies/<track>/<sid>.py`.

---

## 8. Troubleshooting notes

- **`generator raised StopIteration` during a run** — this was a masked error
  where `traceback.format_exc()` itself crashed on Python 3.12 frames compiled
  from `exec()`'d strategy code (a `co_lnotab` metadata bug). The backtest
  error handlers (`BaseRunner._train_backtest`,
  `worker.run_backtest`) now fall back to `Type: message` when the formatter
  fails, so the real error surfaces. If it reappears, check that both handlers
  still guard `format_exc()`.
- **`SymbolNotFound: Symbol 'ES' was not found` on futures** — the coder must
  use `future_symbol("ES")` / `future_symbol("NQ")`, not `symbol(...)` (which
  does not exist on the futures track). The coder system prompt documents this;
  if it regresses, check `src/coders/team.py::SYSTEM_PROMPT`.
- **Empty completions from the implementation model** — `OpenRouterClient.chat`
  returns `""` on an empty completion instead of retrying the identical request;
  the coder re-prompts ("you returned no python code") up to
  `max_refinement_attempts`. Candidates that still fail are marked failed and
  logged, never crash the run.
- **A failed island in a tier** — the run continues (failures are recorded and
  logged via `_log_failed`); placement simply skips failed candidates. To fill a
  specific island, generate one more candidate for that island with the same
  runner machinery (see `_generate_candidate`).
