"""Step 13 tests: PBO via CSCV and Deflated Sharpe Ratio.

Covers the CSCV block construction, the IS-best / OOS-rank / logit pipeline,
the deterministic IS-best tie-break, degenerate cases, and the DSR formula
plus its degenerate cases.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from src.report.backtest_validation import (
    EULER_MASCHERONI,
    cscv_pbo,
    deflated_sharpe,
    _argmax_with_tiebreak,
    _average_rank,
)


# ---------------------------------------------------------------------------
# PBO via CSCV
# ---------------------------------------------------------------------------

def test_cscv_pbo_combination_count_is_12870():
    # identical zero returns: every IS-best choice is a tie, OOS ranks all tie
    # at (N+1)/2 -> omega=0.5 -> lambda=0 -> counted as <= 0.
    returns = np.zeros((160, 4))
    res = cscv_pbo(returns)
    assert res["n_combinations"] == 12870
    assert res["pbo"] == pytest.approx(1.0)
    assert res["undefined"] is None


def test_cscv_pbo_degenerate_n_lt_2():
    res = cscv_pbo(np.zeros((160, 1)))
    assert res["pbo"] is None
    assert res["undefined"] == "N < 2"


def test_cscv_pbo_degenerate_t_lt_s():
    res = cscv_pbo(np.zeros((8, 3)))
    assert res["pbo"] is None
    assert "T=" in res["undefined"]


def test_cscv_pbo_no_oos_information_is_about_half():
    # iid returns: in-sample selection carries no OOS information -> PBO ~ 0.5.
    rng = np.random.default_rng(7)
    returns = rng.normal(size=(320, 8))
    res = cscv_pbo(returns)
    assert res["undefined"] is None
    assert abs(res["pbo"] - 0.5) < 0.15


def test_cscv_pbo_persistent_best_is_zero():
    # strategy 0 strictly best in every block -> always IS-best and OOS-best
    # (rank 1) -> lambda > 0 always -> PBO = 0.
    rng = np.random.default_rng(3)
    returns = rng.normal(size=(320, 5))
    returns[:, 0] += 0.5
    res = cscv_pbo(returns)
    assert res["pbo"] == pytest.approx(0.0)


def test_argmax_tiebreak_prefers_lowest_gen_then_island_then_insertion():
    values = np.array([1.0, 1.0, 1.0, 1.0])
    meta = [(3, 0, 0), (1, 2, 9), (2, 1, 3), (1, 1, 1)]
    # lowest (gen, island, insertion) = (1,1,1) -> index 3
    assert _argmax_with_tiebreak(values, meta) == 3
    # (1,0,5) beats (1,1,1) on island
    meta2 = [(3, 0, 0), (1, 2, 9), (2, 1, 3), (1, 0, 5)]
    assert _argmax_with_tiebreak(values, meta2) == 3
    # equal (gen, island) -> lower insertion wins
    meta3 = [(1, 1, 3), (1, 1, 2), (1, 1, 5), (1, 1, 1)]
    assert _argmax_with_tiebreak(values, meta3) == 3


def test_average_rank_ties():
    vals = np.array([0.5, 0.5, 0.5, 0.1, 0.9])
    # sorted: 0.1(rank1), 0.5,0.5,0.5 (avg rank 3), 0.9(rank5)
    assert _average_rank(vals, 1) == 3.0
    assert _average_rank(vals, 0) == 3.0
    assert _average_rank(vals, 3) == 1.0
    assert _average_rank(vals, 4) == 5.0


# ---------------------------------------------------------------------------
# DSR (Deflated Sharpe Ratio)
# ---------------------------------------------------------------------------

def test_dsr_sigma_trials_zero_is_undefined():
    res = deflated_sharpe(1.0, np.random.default_rng(0).normal(size=200), 100, 0.0)
    assert res["dsr"] is None
    assert "sigma_SR_trials == 0" in res["undefined"]


def test_dsr_matches_spec_formula():
    rng = np.random.default_rng(11)
    test_returns = rng.normal(size=400)
    sr_hat = 1.5
    n_trials = 200
    sigma_trials = 0.4

    g3 = float(np.mean((test_returns - test_returns.mean()) ** 3)
               / test_returns.std(ddof=1) ** 3)
    g4 = (float(np.mean((test_returns - test_returns.mean()) ** 4)
                / test_returns.std(ddof=1) ** 4) - 3.0)
    sigma_hat = np.sqrt((1.0 - g3 * sr_hat
                         + ((g4 - 1.0) / 4.0) * sr_hat ** 2) / (399.0))
    gamma = EULER_MASCHERONI
    n = float(n_trials)
    sr0 = sigma_trials * ((1.0 - gamma) * norm.ppf(1.0 - 1.0 / n)
                          + gamma * norm.ppf(1.0 - 1.0 / (n * np.e)))
    expected = float(norm.cdf((sr_hat - sr0) / sigma_hat))

    res = deflated_sharpe(sr_hat, test_returns, n_trials, sigma_trials)
    assert res["dsr"] == pytest.approx(expected, abs=1e-6)
    assert res["undefined"] is None
    assert res["sigma_trials"] == pytest.approx(sigma_trials)


def test_dsr_monotone_in_sr_hat():
    # t-distributed returns (leptokurtic, like daily returns) keep sigma_hat>0
    rng = np.random.default_rng(5)
    test_returns = rng.standard_t(df=5, size=600)
    low = deflated_sharpe(0.2, test_returns, 100, 0.5)["dsr"]
    high = deflated_sharpe(1.2, test_returns, 100, 0.5)["dsr"]
    assert low is not None and high is not None
    assert 0.0 <= low < high <= 1.0


def test_dsr_very_high_sharpe_is_near_one():
    rng = np.random.default_rng(2)
    test_returns = rng.standard_t(df=5, size=1000)
    res = deflated_sharpe(5.0, test_returns, 50, 0.3)
    assert res["dsr"] is not None and res["dsr"] > 0.99


def test_dsr_trial_count_uses_archive_log_not_failed_records(tmp_path, monkeypatch):
    import csv
    import json
    from types import SimpleNamespace

    import src.report.backtest_validation as bv

    out = tmp_path / "outputs"
    monkeypatch.setattr(bv, "OUTPUTS", out)
    run_dir = out / "evolution" / "equities" / "a" / "tier1"
    (run_dir / "test_returns").mkdir(parents=True)
    rng = np.random.default_rng(31)
    np.savetxt(run_dir / "test_returns" / "winner.csv",
               rng.standard_t(df=5, size=400) * 0.01 + 0.002)
    (run_dir / "summary.json").write_text(json.dumps({"dsr_target": "winner"}))

    with (run_dir / "archive_log.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["entry_id", "combined_score"])
        w.writeheader()
        w.writerow({"entry_id": "winner", "combined_score": "0.1"})
        w.writerow({"entry_id": "other", "combined_score": "0.5"})

    with (run_dir / "records.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["entry_id", "status", "combined_score"])
        w.writeheader()
        w.writerow({"entry_id": "winner", "status": "ok", "combined_score": "0.1"})
        for i in range(10):
            w.writerow({"entry_id": f"failed{i}", "status": "failed", "combined_score": ""})

    res = bv._dsr_for_track(SimpleNamespace(name="equities", bars_per_year=252), "tier1")
    assert res["n_archive_entries"] == 2
    assert res["n_trials"] == 2
    assert res["sigma_trials"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# End-to-end run_validation over a synthetic persisted outputs tree
# ---------------------------------------------------------------------------

def test_run_validation_end_to_end(tmp_path, monkeypatch):
    import json

    import src.report.backtest_validation as bv

    rng = np.random.default_rng(9)
    T = 320
    n = 6
    rets = rng.normal(size=(T, n))
    rets[:, 0] += 0.4  # persistent best

    out = tmp_path / "outputs"
    monkeypatch.setattr(bv, "OUTPUTS", out)
    monkeypatch.setattr(bv, "REPORT_DIR", out / "reports")

    # Path A run dir
    a_dir = out / "evolution" / "equities" / "a" / "tier1"
    (a_dir / "returns").mkdir(parents=True)
    (a_dir / "test_returns").mkdir(parents=True)
    sids = [f"eqa{i}g1c1" for i in range(n)]
    for j, sid in enumerate(sids):
        np.savetxt(a_dir / "returns" / f"{sid}.csv", rets[:, j])
    np.savetxt(a_dir / "test_returns" / f"{sids[0]}.csv", rng.normal(size=400))

    import csv

    with (a_dir / "archive_log.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["entry_id", "generation", "island", "combined_score"])
        w.writeheader()
        for j, sid in enumerate(sids):
            w.writerow({"entry_id": sid, "generation": 1, "island": j,
                        "combined_score": 0.1 + 0.1 * j})
    with (a_dir / "archive_cells.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["entry_id", "generation", "island"])
        w.writeheader()
        for j, sid in enumerate(sids):
            w.writerow({"entry_id": sid, "generation": 1, "island": j})
    with (a_dir / "records.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["entry_id", "combined_score"])
        w.writeheader()
        for j, sid in enumerate(sids):
            w.writerow({"entry_id": sid, "combined_score": 0.1 + 0.1 * j})
    (a_dir / "summary.json").write_text(json.dumps({"dsr_target": sids[0]}))

    # Path B run dir
    b_dir = out / "evolution" / "equities" / "b" / "tier1"
    (b_dir / "returns").mkdir(parents=True)
    bsids = [f"eqb{i}g1c1" for i in range(n)]
    for j, sid in enumerate(bsids):
        np.savetxt(b_dir / "returns" / f"{sid}.csv", rets[:, j])
    with (b_dir / "population_history.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["strategy_id", "generation", "island"])
        w.writeheader()
        for j, sid in enumerate(bsids):
            w.writerow({"strategy_id": sid, "generation": 1, "island": j})
    (b_dir / "summary.json").write_text(json.dumps({"test": {s: {} for s in bsids}}))

    report = bv.run_validation("equities", "tier1")
    pa, pb = report["path_a"], report["path_b"]
    assert pa["pbo"]["pbo"] is not None and 0.0 <= pa["pbo"]["pbo"] <= 1.0
    assert pb["pbo"]["pbo"] is not None and 0.0 <= pb["pbo"]["pbo"] <= 1.0
    assert pa["pbo"]["n_combinations"] == 12870
    assert pa["dsr"]["dsr"] is not None and 0.0 < pa["dsr"]["dsr"] <= 1.0
    assert (out / "reports" / "equities" / "backtest_validation_tier1.json").exists()
