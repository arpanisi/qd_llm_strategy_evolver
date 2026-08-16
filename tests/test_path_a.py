"""Acceptance-criteria tests for Path A (Steps 2A/3/4A/8A/9A).

Covers: frozen bin-edge freezing with headroom, out-of-range clamping, cell
placement with Step 8A's 0.02 IR near-tie turnover tie-break, the Step 4A
parent-selection probability split, diverse-cousin neighbor perturbation with
the resample cap, migration copy semantics (new island-tagged entry), and the
Step 9A Breadth formula.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.taxonomy import StyleVector
from src.path_a.archive import (
    ArchiveEntry,
    ContinuousBins,
    FeatureMap,
    freeze_bin_edges,
)
from src.path_a.sampling import PathASampler
from src.path_a.scoring import better, breadth_from_cells, rank_by_score
from src.config.settings import PathAConfig


# ---------------------------------------------------------------------------
# Step 2A: bin edges frozen once, clamped
# ---------------------------------------------------------------------------

def _seed_df() -> pd.DataFrame:
    return pd.DataFrame({
        "seed": ["seed_0_trend_following"] * 9,
        "Trading Frequency": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
        "Max Drawdown": [-0.5, -0.4, -0.3, -0.2, -0.15, -0.12, -0.10, -0.08, -0.06],
        "Sharpe Ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "Sortino Ratio": [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
        "Total Return": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    })


def test_freeze_bin_edges_expands_observed_range_by_headroom():
    frozen = freeze_bin_edges(_seed_df(), bins_per_dim=16, headroom=0.5)
    tf = frozen["trading_frequency"]
    assert tf.low == pytest.approx(10.0 - 0.5 * 80.0)
    assert tf.high == pytest.approx(90.0 + 0.5 * 80.0)
    assert tf.bins == 16
    assert len(tf.edges) == 17
    assert tf.edges == pytest.approx(tuple(np.linspace(tf.low, tf.high, 17)))


def test_bin_index_clamps_out_of_range_values():
    tf = ContinuousBins("tf", 0.0, 100.0, 10, tuple(range(11)), 0.0, 100.0)
    assert tf.bin_index(-1e9) == 0
    assert tf.bin_index(1e9) == 9
    assert tf.bin_index(55.0) == 5
    assert tf.bin_index(99.9999) == 9


def test_bin_index_handles_nan():
    tf = ContinuousBins("tf", 0.0, 100.0, 10, tuple(range(11)), 0.0, 100.0)
    assert tf.bin_index(float("nan")) == 0


# ---------------------------------------------------------------------------
# Step 8A: Combined Score = IR, turnover tie-break at 0.02 near-tie
# ---------------------------------------------------------------------------

def _entry(entry_id, ir, turnover, style=0, island=0):
    return ArchiveEntry(
        entry_id=entry_id, island=island, generation=1,
        strategy_name=entry_id, style=StyleVector.single(style),
        trading_frequency=50.0, max_drawdown=-0.3, sharpe=0.5, sortino=0.4,
        total_return=0.5, turnover=turnover, information_ratio=ir,
    )


def _fm(ir_near_tie=0.02):
    frozen = freeze_bin_edges(_seed_df(), bins_per_dim=16, headroom=0.5)
    return FeatureMap(frozen, ir_near_tie=ir_near_tie, turnover_tiebreak="lower")


def test_placement_beat_incumbent_on_score():
    fm = _fm()
    ok1, r1 = fm.place(_entry("a", 0.5, 0.4))
    ok2, r2 = fm.place(_entry("b", 0.6, 0.9))
    assert ok1 and "empty" in r1
    assert ok2 and "beat" in r2
    assert list(fm.cells.values())[0].entry_id == "b"


def test_near_tie_lower_turnover_wins_cell():
    fm = _fm(ir_near_tie=0.02)
    fm.place(_entry("a", 0.5, 0.8))
    ok, reason = fm.place(_entry("b", 0.51, 0.2))
    assert ok and "turnover" in reason
    cell_entry = list(fm.cells.values())[0]
    assert cell_entry.entry_id == "b"


def test_near_tie_higher_turnover_loses_cell():
    fm = _fm(ir_near_tie=0.02)
    fm.place(_entry("a", 0.5, 0.2))
    ok, reason = fm.place(_entry("b", 0.51, 0.9))
    assert not ok and "turnover" in reason
    assert list(fm.cells.values())[0].entry_id == "a"


def test_better_and_rank_apply_tiebreak():
    a = _entry("a", 0.5, 0.9)
    b = _entry("b", 0.51, 0.1)
    assert better(a, b, ir_near_tie=0.02).entry_id == "b"
    ranked = rank_by_score([a, b], ir_near_tie=0.02)
    assert [e.entry_id for e in ranked] == ["b", "a"]


def test_rejected_entry_still_in_archive_log():
    fm = _fm()
    fm.place(_entry("a", 0.5, 0.2))
    fm.place(_entry("b", 0.3, 0.1))
    assert len(fm.log) == 2
    assert any(e.entry_id == "b" for e in fm.log)


# ---------------------------------------------------------------------------
# Step 9A: Breadth from occupied cells
# ---------------------------------------------------------------------------

def test_breadth_single_style_is_one():
    fm = _fm()
    for i in range(4):
        fm.place(_entry(f"e{i}", 0.5 + i * 0.3, 0.1, style=0, island=i))
    b = breadth_from_cells(fm)
    assert b["breadth"] == pytest.approx(1.0)
    assert b["distribution"]["10000000"] == pytest.approx(1.0)


def test_breadth_two_even_styles_is_two():
    fm = _fm()
    fm.place(_entry("e0", 0.5, 0.1, style=0))
    fm.place(_entry("e1", 0.6, 0.1, style=1))
    b = breadth_from_cells(fm)
    assert b["breadth"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Step 4A: parent selection probability split + diverse cousins
# ---------------------------------------------------------------------------

def _populated_fm():
    fm = _fm()
    for i in range(9):
        fm.place(_entry(f"seed_{i}:g0", 0.1 + i * 0.05, 0.2, style=min(i, 7), island=i))
    for g in range(1, 3):
        for i in range(9):
            fm.place(_entry(f"x{i}g{g}", 0.3, 0.1, style=i % 2, island=i))
    return fm


def _cfg():
    return PathAConfig(
        alpha=0.5, n_best_cousins=2, n_diverse_cousins=3, n_random_cousins=2,
        cousin_neighbor_sigma=1.0, bitflips=2, resample_attempts=10,
        bins_per_dim=16, bin_headroom=0.5, ir_near_tie=0.02,
        turnover_tiebreak="lower",
    )


def test_parent_selection_respects_probability_split():
    rng = np.random.default_rng(42)
    fm = _populated_fm()
    island = 3
    occupants = [e for e in fm.log if e.island == island and e.placed]
    total = [e for e in fm.log if e.island == island]
    sampler = PathASampler(fm, _cfg(), rng)
    draws = [sampler.sample_parent(island).entry_id for _ in range(4000)]
    # P(occupant) = alpha/|M| + (1-alpha)/|I|
    expected_p_occ = 0.5 / len(occupants) + 0.5 / len(total)
    observed_p_occ = sum(1 for d in draws if any(e.entry_id == d and e.placed for e in total)) / len(draws)
    assert observed_p_occ == pytest.approx(expected_p_occ, abs=0.05)


def test_sample_cousins_count_and_early_gen_fallback():
    rng = np.random.default_rng(7)
    fm = _populated_fm()
    sampler = PathASampler(fm, _cfg(), rng)
    parent = sampler.sample_parent(3)
    cousins = sampler.sample_cousins(3, parent)
    assert len(cousins) == 7
    # with-replacement fallback: an island with exactly 1 entry can still sample
    fm2 = FeatureMap(fm.frozen, ir_near_tie=0.02, turnover_tiebreak="lower")
    fm2.place(_entry("only", 0.2, 0.1, style=3, island=3))
    s2 = PathASampler(fm2, _cfg(), np.random.default_rng(1))
    p2 = s2.sample_parent(3)
    assert s2.sample_cousins(3, p2)  # returns 7 even with one entry


def test_diverse_cousin_neighbor_perturbation_and_cap():
    rng = np.random.default_rng(11)
    fm = _populated_fm()
    sampler = PathASampler(fm, _cfg(), rng)
    parent = next(e for e in fm.log if e.entry_id == "seed_3:g0")
    cousins = sampler._diverse_cousins(parent, [e for e in fm.log if e.island == 3], 3)
    assert len(cousins) == 3
    # all cousins must be drawn from the island's own population
    assert all(c.island == 3 for c in cousins)


# ---------------------------------------------------------------------------
# Step 3: migration copy semantics (Path A clause)
# ---------------------------------------------------------------------------

def test_archive_entry_style_binary_roundtrip():
    v = StyleVector(True, False, True, False, False, False, False, False)
    assert StyleVector.from_binary_string(v.as_binary_string()) == v


# ---------------------------------------------------------------------------
# Step 13: DSR target selection (Path A)
# ---------------------------------------------------------------------------

def test_select_dsr_target_picks_highest_validation_score():
    from src.evolution.path_a_run import _select_dsr_target

    best = {0: "s0", 1: "s1", 2: "s2"}
    val = {"s0": 0.1, "s1": 0.5, "s2": 0.3}
    assert _select_dsr_target(best, val) == "s1"


def test_select_dsr_target_skips_nan_scores():
    from src.evolution.path_a_run import _select_dsr_target

    best = {0: "s0", 1: "s1"}
    val = {"s0": 0.4, "s1": float("nan")}
    assert _select_dsr_target(best, val) == "s0"


def test_select_dsr_target_none_when_all_nan():
    from src.evolution.path_a_run import _select_dsr_target

    best = {0: "s0", 1: "s1"}
    val = {"s0": float("nan"), "s1": float("nan")}
    assert _select_dsr_target(best, val) is None


def test_select_dsr_target_tie_prefers_lowest_island():
    from src.evolution.path_a_run import _select_dsr_target

    best = {0: "s0", 1: "s1"}
    val = {"s0": 0.4, "s1": 0.4}
    assert _select_dsr_target(best, val) == "s0"
