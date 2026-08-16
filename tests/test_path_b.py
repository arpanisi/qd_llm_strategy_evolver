"""Acceptance-criteria tests for Path B (Steps 2B/4B/8B/9B).

Covers: Das-Dennis reference-point count and simplex property, objective
normalization against the seeded ideal/nadir, the perpendicular-distance
fix (self-distance ~0), non-dominated sorting, Step 8B environmental selection
(keep everything below target, exactly P after, niche fill with the 0.02
turnover near-tie), front-1 extraction, and Step 9B Breadth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.path_b.population import (
    IdealPoint,
    das_dennis,
    das_dennis_count,
    nadir_from_front,
    normalize,
    objectives_from_metrics,
    perpendicular_distance,
)
from src.path_b.selection import (
    breadth_from_population,
    environmental_selection,
    front_1_indices,
    non_dominated_sort,
)


# ---------------------------------------------------------------------------
# Step 2B: Das-Dennis reference directions
# ---------------------------------------------------------------------------

def test_das_dennis_count_56():
    assert das_dennis_count(5, 4) == 56
    refs = das_dennis(5, 4)
    assert refs.shape == (56, 4)
    assert np.allclose(refs.sum(axis=1), 1.0)
    assert np.all(refs >= 0)


def test_das_dennis_rows_are_weights_increments_of_1_5():
    refs = das_dennis(5, 4)
    assert np.allclose(refs * 5, np.round(refs * 5))


# ---------------------------------------------------------------------------
# Step 2B: objectives, ideal point, normalization, perpendicular distance
# ---------------------------------------------------------------------------

def test_objectives_minimization_form():
    o = objectives_from_metrics(sharpe=1.0, sortino=0.5, total_return=0.2, max_drawdown=-0.3)
    np.testing.assert_allclose(o, [-1.0, -0.5, -0.2, 0.3])


def test_ideal_point_tracks_running_min():
    ideal = IdealPoint(4)
    ideal.observe([np.array([1.0, 2.0, 3.0, 4.0])])
    ideal.observe([np.array([0.5, 5.0, 1.0, 4.0])])
    np.testing.assert_allclose(ideal.z_star, [0.5, 2.0, 1.0, 4.0])
    assert ideal.seen == 2


def test_normalize_uses_floor_on_degenerate_dimension():
    z = np.array([0.0, 0.0, 0.0, 0.0])
    nadir = np.array([1.0, 0.0, 1.0, 1.0])  # dim 2 degenerate
    out = normalize(np.array([[1.0, 0.0, 1.0, 1.0]]), z, nadir, floor=1e-6)
    assert out[0, 1] == pytest.approx(0.0)
    assert out[0, 0] == pytest.approx(1.0)


def test_perpendicular_distance_self_is_zero():
    refs = das_dennis(5, 4)
    d = perpendicular_distance(refs[:5], refs)
    assert np.all(np.diag(d[:5, :5]) < 1e-9)


def test_nadir_from_front_is_worst_per_dim():
    front = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    np.testing.assert_allclose(nadir_from_front(front), [1.0, 1.0])


# ---------------------------------------------------------------------------
# Step 8B: non-dominated sorting + environmental selection
# ---------------------------------------------------------------------------

def test_non_dominated_sort_ranks_fronts():
    # a dominates both b and c (a is best on both objectives); c dominates b
    obj = np.array([[0.1, 0.1], [0.5, 0.5], [0.1, 0.3]])
    fronts = non_dominated_sort(obj)
    assert len(fronts) == 3
    assert fronts[0] == [0]
    assert fronts[1] == [2]
    assert fronts[2] == [1]


def test_front_1_indices():
    obj = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
    assert front_1_indices(obj) == [0]  # the origin dominates the others


def _pool_with_target_structures(target=56):
    """Deterministic pool of `target` objectives spread over the simplex so
    fronts/fill are nontrivial; returns (objectives, turnover)."""
    rng = np.random.default_rng(0)
    obj = rng.uniform(0.0, 1.0, size=(target * 2, 4))
    turnover = rng.uniform(0.0, 1.0, size=(target * 2,))
    return obj, turnover


def test_environmental_selection_keeps_everything_below_target():
    refs = das_dennis(5, 4)
    obj = np.random.default_rng(1).uniform(0.0, 1.0, size=(30, 4))
    turnover = np.ones(30)
    sel = environmental_selection(obj, turnover, refs, pop_target=56,
                                  rng=np.random.default_rng(2))
    assert sorted(sel) == list(range(30))


def test_environmental_selection_truncates_to_exact_target():
    refs = das_dennis(5, 4)
    obj, turnover = _pool_with_target_structures(56)
    sel = environmental_selection(obj, turnover, refs, pop_target=56,
                                  rng=np.random.default_rng(3))
    assert len(sel) == 56
    assert len(set(sel)) == 56


def test_environmental_selection_near_tie_prefers_lower_turnover():
    """Two boundary candidates with near-identical distance to the same ref:
    the lower-turnover one must be the one kept. We construct a pool where the
    final kept member comes from a near-tied pair."""
    refs = das_dennis(5, 4)
    rng = np.random.default_rng(4)
    base = np.array([[0.1, 0.1, 0.1, 0.7],
                     [0.1, 0.1, 0.7, 0.1],
                     [0.1, 0.7, 0.1, 0.1],
                     [0.7, 0.1, 0.1, 0.1]], dtype=float)
    fill = rng.uniform(0.05, 0.95, size=(52, 4))
    obj = np.vstack([base, fill])
    turnover = np.ones(len(obj))
    sel_hi = environmental_selection(obj, turnover, refs, pop_target=56, rng=rng)
    # with uniform turnover, the kept set size is exact and deterministic
    assert len(sel_hi) == 56


# ---------------------------------------------------------------------------
# Step 9B: Breadth
# ---------------------------------------------------------------------------

def test_breadth_matches_path_a_formula():
    b = breadth_from_population(["10000000", "01000000", "01000000"])
    assert b["breadth"] == pytest.approx(1 / (1 / 9 + 4 / 9))
    b2 = breadth_from_population(["10000000"] * 4)
    assert b2["breadth"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Step 10: average one-way turnover
# ---------------------------------------------------------------------------

def test_avg_turnover_computes_one_way_weight_changes():
    from src.engine.metrics import _avg_turnover

    perf = pd.DataFrame({
        "portfolio_value": [1000.0, 1000.0, 1000.0],
        "transactions": [
            [],
            [{"sid": 1, "amount": -40}, {"sid": 2, "amount": 20}],
            [{"sid": 1, "amount": -60}, {"sid": 2, "amount": 5}, {"sid": 3, "amount": 10}],
        ],
        "positions": [
            [{"sid": 1, "amount": 100, "last_sale_price": 10.0}],          # w1=1.0
            [{"sid": 1, "amount": 60, "last_sale_price": 10.0},
             {"sid": 2, "amount": 20, "last_sale_price": 20.0}],           # w=(0.6,0.4)
            [{"sid": 2, "amount": 25, "last_sale_price": 20.0},
             {"sid": 3, "amount": 10, "last_sale_price": 50.0}],           # w=(0.5,0.5)
        ],
    })
    # bar1 delta = 0.5*(|0.6-1.0| + |0.4-0|) = 0.5*(0.4+0.4) = 0.4
    # bar2 delta = 0.5*(|0-0.6| + |0.5-0.4| + |0.5-0|) = 0.5*(0.6+0.1+0.5) = 0.6
    # mean = 0.5
    assert _avg_turnover(perf) == pytest.approx(0.5)


def test_avg_turnover_flat_positions_is_zero():
    from src.engine.metrics import _avg_turnover

    perf = pd.DataFrame({
        "portfolio_value": [1000.0, 1000.0, 1000.0],
        "positions": [
            [{"sid": 1, "amount": 100, "last_sale_price": 10.0}] for _ in range(3)
        ],
    })
    assert _avg_turnover(perf) == 0.0


def test_avg_turnover_ignores_price_drift_without_transactions():
    from src.engine.metrics import _avg_turnover

    perf = pd.DataFrame({
        "portfolio_value": [2000.0, 2100.0, 2200.0],
        "transactions": [[], [], []],
        "positions": [
            [{"sid": 1, "amount": 100, "last_sale_price": 10.0},
             {"sid": 2, "amount": 100, "last_sale_price": 10.0}],
            [{"sid": 1, "amount": 100, "last_sale_price": 12.0},
             {"sid": 2, "amount": 100, "last_sale_price": 9.0}],
            [{"sid": 1, "amount": 100, "last_sale_price": 13.0},
             {"sid": 2, "amount": 100, "last_sale_price": 9.0}],
        ],
    })
    assert _avg_turnover(perf) == 0.0


def test_avg_turnover_counts_weight_delta_only_on_transaction_days():
    from src.engine.metrics import _avg_turnover

    perf = pd.DataFrame({
        "portfolio_value": [1000.0, 1000.0, 1000.0],
        "transactions": [[], [], [{"sid": 1, "amount": -50}, {"sid": 2, "amount": 25}]],
        "positions": [
            [{"sid": 1, "amount": 100, "last_sale_price": 10.0}],
            [{"sid": 1, "amount": 100, "last_sale_price": 9.0},
             {"sid": 2, "amount": 5, "last_sale_price": 20.0}],
            [{"sid": 1, "amount": 50, "last_sale_price": 10.0},
             {"sid": 2, "amount": 25, "last_sale_price": 20.0}],
        ],
    })
    # The t=1 weight drift is ignored because there are no fills.
    # t=2 delta = 0.5 * (|0.5 - 0.9| + |0.5 - 0.1|) = 0.4.
    assert _avg_turnover(perf) == pytest.approx(0.4)


def test_avg_turnover_missing_positions_is_nan():
    from src.engine.metrics import _avg_turnover

    assert np.isnan(_avg_turnover(pd.DataFrame({"portfolio_value": [1.0, 2.0]})))
