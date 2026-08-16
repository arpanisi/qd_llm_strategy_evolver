"""Acceptance-criteria tests for the Step 4B mating selection and Step 12
path-comparison dominance check (pure logic, no LLM/Zipline)."""

from __future__ import annotations

import numpy as np
import pytest

from src.path_b.mating import PathBMating
from src.path_b.population import (
    ReferencePointTable,
    das_dennis,
    objectives_from_metrics,
)
from src.report.comparison import dominates


# ---------------------------------------------------------------------------
# Step 4B: binary tournament + nearest reference-direction cousins
# ---------------------------------------------------------------------------

class _Ind:
    def __init__(self, sid, objectives):
        self.strategy_id = sid
        self.objectives = np.asarray(objectives, dtype=float)

    def metric(self, key):
        return 0.1


def _population_and_table(n=20):
    rng = np.random.default_rng(0)
    inds = [_Ind(f"s{i}", rng.uniform(0.0, 1.0, size=4)) for i in range(n)]
    refs = das_dennis(5, 4)
    norm = np.asarray([i.objectives for i in inds])
    table = ReferencePointTable(refs)
    ids = [i.strategy_id for i in inds]
    table.recompute(ids, norm)
    return inds, norm, table, refs


def test_mating_selects_parent_and_seven_cousins():
    inds, norm, table, refs = _population_and_table()
    mating = PathBMating(refs, np.random.default_rng(1))
    parent, cousins = mating.sample_parent_and_cousins(
        inds, norm, table,
        {ind.strategy_id: idx for idx, ind in enumerate(inds)},
    )
    assert parent in inds
    assert len(cousins) == 7
    assert all(c in inds for c in cousins)


def test_mating_early_gen_with_replacement():
    inds, norm, table, refs = _population_and_table(n=1)
    mating = PathBMating(refs, np.random.default_rng(2))
    parent, cousins = mating.sample_parent_and_cousins(
        inds, norm, table, {ind.strategy_id: 0 for ind in inds}
    )
    assert parent is inds[0]
    assert len(cousins) == 7


# ---------------------------------------------------------------------------
# Step 12: dominance check
# ---------------------------------------------------------------------------

def test_dominates_requires_strictly_better_on_one():
    a = np.array([0.5, 0.5, 0.5, 0.5])
    b = np.array([0.6, 0.5, 0.5, 0.5])
    assert dominates(b, a)
    assert not dominates(a, b)
    assert not dominates(np.array([0.5, 0.5, 0.5, 0.5]), a)  # equal: no strict better


def test_dominates_needs_at_least_as_good_on_all():
    a = np.array([0.5, 0.9, 0.5, 0.5])
    b = np.array([0.6, 0.8, 0.5, 0.5])
    assert not dominates(b, a)  # worse on objective 2
