"""Step 13 — Backtest-Validation: Multiple-Testing-Corrected Significance.

Two read-only diagnostics computed from a completed tier's persisted data
(no re-backtest, no change to Steps 1-12 selection/scoring/reporting):

* PBO (Probability of Backtest Overfitting) via Combinatorially Symmetric
  Cross-Validation (CSCV) with S=16 contiguous blocks (C(16,8)=12,870 block
  splits). Scope: Path A = every occupied feature-map cell's entry; Path B =
  the final validation-confirmed Pareto set. Inputs are the strategies'
  training-window daily returns (persisted during the run) plus each entry's
  (generation, island, insertion-order) for the deterministic IS-best
  tie-break.

* DSR (Deflated Sharpe Ratio), Path A only: tests whether the track's single
  best strategy's test-window Sharpe is still significant after accounting for
  the number of trials that produced it. The target is the per-island-best
  strategy (already test-rerun by Step 8A) with the highest validation-window
  Combined Score, so its test-window daily returns are already persisted — no
  additional backtest.

Both follow the locked spec in the coding plan (Step 13). Degenerate cases
report "undefined" with a reason rather than a fabricated number.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import kurtosis, norm, skew

from src.config.settings import load_config

OUTPUTS = Path(__file__).resolve().parents[2] / "outputs"
REPORT_DIR = OUTPUTS / "reports"

EULER_MASCHERONI = 0.5772156649015329
S_BLOCKS = 16


# ---------------------------------------------------------------------------
# PBO via CSCV
# ---------------------------------------------------------------------------

def _average_rank(values: np.ndarray, idx: int) -> float:
    """1-based rank of values[idx] among all values (ascending; rank 1 =
    worst), with ties assigned the average of the ranks they jointly occupy."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return float(ranks[idx])


def _argmax_with_tiebreak(values: np.ndarray, meta: Optional[list]) -> int:
    """Index of the maximum value; ties broken by lowest (generation, island,
    insertion-order) tuple per the Step 13 spec (deterministic, required for
    bit-reproducibility), else the earliest index."""
    mx = float(values.max())
    idxs = np.flatnonzero(values == mx)
    if len(idxs) == 1 or meta is None:
        return int(idxs[0])
    best = min(idxs, key=lambda i: (meta[i][0], meta[i][1], meta[i][2]))
    return int(best)


def cscv_pbo(returns: np.ndarray, meta: Optional[list] = None,
             s_blocks: int = S_BLOCKS) -> dict:
    """returns: (T, N) training-window daily returns, identical T for all N
    strategies. meta: per-column (generation, island, insertion_order).
    Returns {'pbo': float in [0,1], 'n_combinations': int, 'undefined': reason|None}."""
    T, N = returns.shape
    if N < 2:
        return {"pbo": None, "n_combinations": 0, "undefined": "N < 2"}
    if T < s_blocks:
        return {"pbo": None, "n_combinations": 0,
                "undefined": f"T={T} < S={s_blocks}"}

    base, rem = divmod(T, s_blocks)
    sizes = [base + (1 if i < rem else 0) for i in range(s_blocks)]
    bounds = []
    start = 0
    for sz in sizes:
        bounds.append((start, start + sz))
        start += sz

    s1 = np.zeros((s_blocks, N))
    s2 = np.zeros((s_blocks, N))
    n = np.zeros(s_blocks)
    for b, (lo, hi) in enumerate(bounds):
        seg = returns[lo:hi]
        n[b] = seg.shape[0]
        s1[b] = seg.sum(axis=0)
        s2[b] = (seg ** 2).sum(axis=0)

    total_n = float(T)
    total_s1 = returns.sum(axis=0)
    total_s2 = (returns ** 2).sum(axis=0)

    def sharpe_vec(s, ss, ncnt):
        mean = s / ncnt
        var = np.maximum(ss / ncnt - mean ** 2, 1e-12)
        return mean / np.sqrt(var)

    lam_le_zero = 0
    n_comb = 0
    half = s_blocks // 2
    for is_blocks in combinations(range(s_blocks), half):
        is_blocks = np.asarray(is_blocks, dtype=int)
        mask = np.zeros(s_blocks, dtype=bool)
        mask[is_blocks] = True
        is_n = float(n[mask].sum())
        is_s1 = s1[mask].sum(axis=0)
        is_s2 = s2[mask].sum(axis=0)
        is_sr = sharpe_vec(is_s1, is_s2, is_n)
        best = _argmax_with_tiebreak(is_sr, meta)

        oos_sr = sharpe_vec(total_s1 - is_s1, total_s2 - is_s2, total_n - is_n)
        rank = _average_rank(oos_sr, best)
        omega = rank / (N + 1.0)
        lam = float(np.log(omega / (1.0 - omega)))
        if lam <= 0.0:
            lam_le_zero += 1
        n_comb += 1

    return {"pbo": lam_le_zero / n_comb, "n_combinations": n_comb,
            "undefined": None}


# ---------------------------------------------------------------------------
# DSR (Deflated Sharpe Ratio) — Path A only
# ---------------------------------------------------------------------------

def deflated_sharpe(sr_hat: float, test_returns: np.ndarray, n_trials: int,
                    sigma_trials: float) -> dict:
    """DSR = Phi((SR_hat - SR_0) / sigma_hat_SR) per the Step 13 formula.
    Returns the DSR plus every intermediate so the computation is auditable."""
    T = len(test_returns)
    if n_trials < 2 or T < 2:
        return {"dsr": None, "sr_hat": None, "sr0": None, "sigma_hat": None,
                "n_trials": n_trials, "sigma_trials": sigma_trials,
                "undefined": "insufficient data (n_trials<2 or T<2)"}
    if not np.isfinite(sigma_trials) or sigma_trials <= 0.0:
        return {"dsr": None, "sr_hat": sr_hat, "sr0": None, "sigma_hat": None,
                "n_trials": n_trials, "sigma_trials": sigma_trials,
                "undefined": "sigma_SR_trials == 0 (no cross-trial IR spread)"}

    g3 = float(skew(test_returns))
    g4 = float(kurtosis(test_returns))  # excess kurtosis
    sigma_hat = np.sqrt((1.0 - g3 * sr_hat + ((g4 - 1.0) / 4.0) * sr_hat ** 2)
                        / (T - 1.0))
    if not np.isfinite(sigma_hat) or sigma_hat <= 0.0:
        return {"dsr": None, "sr_hat": sr_hat, "sr0": None,
                "sigma_hat": float(sigma_hat), "n_trials": n_trials,
                "sigma_trials": sigma_trials,
                "undefined": "sigma_hat_SR not positive/finite"}

    gamma = EULER_MASCHERONI
    n = float(n_trials)
    sr0 = sigma_trials * ((1.0 - gamma) * norm.ppf(1.0 - 1.0 / n)
                          + gamma * norm.ppf(1.0 - 1.0 / (n * np.e)))
    dsr = float(norm.cdf((sr_hat - sr0) / sigma_hat))
    return {"dsr": dsr, "sr_hat": sr_hat, "sr0": float(sr0),
            "sigma_hat": float(sigma_hat), "n_trials": n_trials,
            "sigma_trials": float(sigma_trials), "T": int(T), "undefined": None}


# ---------------------------------------------------------------------------
# Loading persisted run data
# ---------------------------------------------------------------------------

def _load_returns(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    return np.loadtxt(path)


def _train_returns_for(sid: str, run_dir: Path, search_dir: Path) -> Optional[np.ndarray]:
    r = _load_returns(run_dir / "returns" / f"{sid}.csv")
    if r is None:
        r = _load_returns(search_dir / "returns" / f"{sid}.csv")
    return r


def _path_a_scope(track: str, tier: str) -> dict:
    """Occupied feature-map cells (one entry per cell) with metadata."""
    run_dir = OUTPUTS / "evolution" / track / "a" / tier
    cells_csv = run_dir / "archive_cells.csv"
    log_csv = run_dir / "archive_log.csv"
    entries = []
    meta = {}
    if cells_csv.exists():
        cells = _read_csv(cells_csv)
        log = _read_csv(log_csv) if log_csv.exists() else []
        log_meta = {row["entry_id"]: (int(row["generation"]), int(row["island"]), i)
                    for i, row in enumerate(log)}
        for row in cells:
            sid = row["entry_id"]
            entries.append(sid)
            meta[sid] = log_meta.get(sid, (0, 0, 0))
    return {"entries": entries, "meta": meta, "run_dir": run_dir}


def _path_b_scope(track: str, tier: str) -> dict:
    """Final validation-confirmed Pareto set, combined across islands."""
    run_dir = OUTPUTS / "evolution" / track / "b" / tier
    hist_csv = run_dir / "population_history.csv"
    summary_csv = run_dir / "summary.json"
    entries = []
    meta = {}
    if hist_csv.exists():
        hist = _read_csv(hist_csv)
        meta = {row["strategy_id"]: (int(row["generation"]), int(row["island"]), i)
                for i, row in enumerate(hist)}
    if summary_csv.exists():
        summary = json.loads(summary_csv.read_text())
        entries = list(summary.get("test", {}).keys())
    return {"entries": entries, "meta": meta, "run_dir": run_dir}


def _read_csv(path: Path) -> list[dict]:
    import csv

    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _pbo_for_scope(scope: dict, search_dir: Path, track_cfg) -> dict:
    cols, meta_list = [], []
    for sid in scope["entries"]:
        rets = _train_returns_for(sid, scope["run_dir"], search_dir)
        if rets is None:
            continue
        cols.append(rets)
        meta_list.append(scope["meta"].get(sid, (0, 0, 0)))
    if not cols:
        return {"pbo": None, "n_combinations": 0,
                "undefined": "no persisted returns for scope entries",
                "n_strategies": 0}
    T = cols[0].shape[0]
    if any(c.shape[0] != T for c in cols):
        return {"pbo": None, "n_combinations": 0,
                "undefined": "inconsistent returns length across scope",
                "n_strategies": len(cols)}
    matrix = np.column_stack(cols)
    result = cscv_pbo(matrix, meta_list)
    result["n_strategies"] = matrix.shape[1]
    return result


def _dsr_for_track(track_cfg, tier: str) -> dict:
    track = track_cfg.name
    run_dir = OUTPUTS / "evolution" / track / "a" / tier
    summary_csv = run_dir / "summary.json"
    archive_log_csv = run_dir / "archive_log.csv"
    if not summary_csv.exists() or not archive_log_csv.exists():
        return {"dsr": None, "undefined": "missing summary/archive log for Path A",
                "track": track}
    summary = json.loads(summary_csv.read_text())
    target = summary.get("dsr_target")
    if not target:
        return {"dsr": None, "undefined": "no dsr_target in Path A summary",
                "track": track}

    test_returns = _load_returns(run_dir / "test_returns" / f"{target}.csv")
    if test_returns is None or test_returns.shape[0] < 2:
        return {"dsr": None, "sr_hat": None, "undefined":
                f"no test-window returns for target {target}", "track": track}

    sr_hat = (np.sqrt(track_cfg.bars_per_year)
              * test_returns.mean() / test_returns.std(ddof=0))
    if not np.isfinite(sr_hat):
        return {"dsr": None, "sr_hat": None, "undefined": "test Sharpe not finite",
                "track": track}

    archive_records = _read_csv(archive_log_csv)
    n_trials = len(archive_records)
    irs = [float(r["combined_score"]) for r in archive_records
           if r.get("combined_score", "") not in ("", "nan", "NaN")]
    sigma_trials = float(np.std(irs)) if len(irs) >= 2 else 0.0

    result = deflated_sharpe(sr_hat, test_returns, n_trials, sigma_trials)
    result["track"] = track
    result["target"] = target
    result["n_archive_entries"] = n_trials
    return result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_validation(track_name: str, tier: str) -> dict:
    cfg = load_config()
    track = next(t for t in cfg.tracks if t.name == track_name)
    search_dir = OUTPUTS / "search" / track_name

    path_a_scope = _path_a_scope(track_name, tier)
    path_b_scope = _path_b_scope(track_name, tier)

    report = {
        "track": track_name,
        "tier": tier,
        "path_a": {
            "pbo": _pbo_for_scope(path_a_scope, search_dir, track),
            "dsr": _dsr_for_track(track, tier),
        },
        "path_b": {
            "pbo": _pbo_for_scope(path_b_scope, search_dir, track),
        },
    }

    out = REPORT_DIR / track_name
    out.mkdir(parents=True, exist_ok=True)
    (out / f"backtest_validation_{tier}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True))
    return report
