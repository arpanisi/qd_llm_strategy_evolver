"""Tests for the concurrency refactor (src.evolution.worker + run() scheduler).

Covers the plan's concurrency model as implemented:
  * worker purity — Steps 5/6/7 run as a free function over a self-contained,
    picklable task dict; the worker never imports the orchestrator and never
    touches fm/population/ref-table state;
  * the shared produce_candidate_core pipeline (ok path + each failure path);
  * the run() scheduler — islands dispatched concurrently to a process pool,
    results collected in island order, placement single-threaded, and every
    island's candidate lands in the archive (Path A).
"""

from __future__ import annotations

import inspect
import multiprocessing
import pickle
from pathlib import Path

import numpy as np
import pytest

import src.evolution.worker as worker
from src.config.settings import load_config
from src.data.taxonomy import StyleVector
from src.evolution.orchestrator import tier_specs
from src.evolution.path_a_run import PathARunner
from src.evolution.record import StrategyRecord
from src.roles.schemas import Evaluation, Hypothesis


# ---------------------------------------------------------------------------
# Worker purity contract (the plan's concurrency model)
# ---------------------------------------------------------------------------

def test_worker_never_imports_orchestrator():
    import ast

    tree = ast.parse(inspect.getsource(worker))
    imported = {
        (getattr(n, "module", None), n.name if hasattr(n, "name") else n.names[0].name)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Import, ast.ImportFrom))
    }
    assert not any(m == "src.evolution.orchestrator" for m, _ in imported)
    assert not any("orchestrator" in (m or "") for m, _ in imported)


def test_task_dict_is_self_contained_and_picklable():
    cfg = load_config()
    track = cfg.tracks[0]
    runner = PathARunner(cfg, track, "a", tier_specs()["tier1"])
    task = runner.candidate_factory.task_inputs(2, 1, 1)

    forbidden = {"fm", "population", "ref_table", "islands", "records",
                 "codes", "runner", "insight_log"}
    assert not (set(task.keys()) & forbidden)
    assert task["parent"] is not None and len(task["cousins"]) == 7
    assert isinstance(task["insights"], list)
    pickle.loads(pickle.dumps(task))


# ---------------------------------------------------------------------------
# Shared Steps 5/6/7 pipeline (produce_candidate_core)
# ---------------------------------------------------------------------------

def _task():
    cfg = load_config()
    track = cfg.tracks[0]
    runner = PathARunner(cfg, track, "a", tier_specs()["tier1"])
    return runner.candidate_factory.task_inputs(2, 1, 1)


def _ok_backtest():
    from src.coders.team import BacktestResult

    return BacktestResult(
        ok=True,
        metrics={
            "sharpe": 0.5, "sortino": 0.6, "ir": 0.3, "mdd": -0.2,
            "total_return": 0.4, "turnover": 0.1, "trading_frequency": 50.0,
            "n_fills": 120, "n_days": 100, "final_equity": 1.0e6,
        },
        info={},
    )


class _StubResearch:
    def generate_hypothesis(self, parent, cousins, insights, track):
        return Hypothesis("h", "r", "o", "e", "rl", "n")


class _StubCoder:
    def __init__(self, result):
        self.result = result

    def generate(self, sid, hypothesis, parent, cousins, track):
        return "def initialize(context):\n    pass\n", self.result


class _StubEvaluator:
    def evaluate(self, record, backtest_log, baseline_summary):
        return Evaluation(
            hypothesis_score=0.6, hypothesis_reasoning="r",
            code_alignment_score=0.8, code_reasoning="r",
            results_score=0.5, results_reasoning="r",
            style_categories=["trend_following", "volatility"],
            insight="keep testing",
        )


def test_core_ok_path_builds_full_record():
    rec = worker.produce_candidate_core(
        _task(), _StubResearch(), _StubCoder(_ok_backtest()), _StubEvaluator())
    assert rec.status == "ok"
    assert rec.metrics["ir"] == 0.3
    assert rec.combined_score == 0.3
    assert rec.objectives is not None and rec.objectives.shape == (4,)
    assert rec.evaluation is not None
    assert rec.style.as_binary_string() == "10100000"


def test_core_coder_failed_backtest_marks_failed():
    from src.coders.team import BacktestResult

    rec = worker.produce_candidate_core(
        _task(), _StubResearch(),
        _StubCoder(BacktestResult(ok=False, error="boom", code="x")),
        _StubEvaluator())
    assert rec.status == "failed"
    assert "boom" in rec.failure_error


def test_core_research_exception_marks_failed():
    class Boom:
        def generate_hypothesis(self, parent, cousins, insights, track):
            raise RuntimeError("research died")

    rec = worker.produce_candidate_core(_task(), Boom(), _StubCoder(_ok_backtest()), _StubEvaluator())
    assert rec.status == "failed"
    assert rec.failure_error.startswith("Step 5 research failed")


def test_core_coder_exception_marks_failed():
    class Boom:
        def generate(self, sid, hypothesis, parent, cousins, track):
            raise RuntimeError("coder died")

    rec = worker.produce_candidate_core(_task(), _StubResearch(), Boom(), _StubEvaluator())
    assert rec.status == "failed"
    assert rec.failure_error.startswith("Step 6 coder failed")


def test_core_evaluation_exception_marks_failed():
    class Boom:
        def evaluate(self, record, backtest_log, baseline_summary):
            raise RuntimeError("eval died")

    rec = worker.produce_candidate_core(_task(), _StubResearch(), _StubCoder(_ok_backtest()), Boom())
    assert rec.status == "failed"
    assert rec.failure_error.startswith("Step 7 evaluation failed")


# ---------------------------------------------------------------------------
# run() scheduler: concurrent dispatch, ordered collection, single-threaded
# placement (Path A archive)
# ---------------------------------------------------------------------------

def _fake_producer(task):
    sid = task["sid"]
    rec = StrategyRecord(
        strategy_id=sid, track=task["track"].name, path=task["path"],
        island=task["island"], generation=task["generation"],
        hypothesis=Hypothesis("h", "r", "o", "e", "rl", "n"), code="pass",
    )
    rec.metrics = {
        "sharpe": 0.5, "sortino": 0.6, "ir": 0.3, "mdd": -0.2,
        "total_return": 0.4, "turnover": 0.1, "trading_frequency": 50.0,
        "n_fills": 120, "n_days": 100, "final_equity": 1.0e6,
    }
    rec.combined_score = 0.3
    rec.objectives = np.array([-0.5, -0.6, -0.4, 0.2])
    rec.evaluation = Evaluation(
        hypothesis_score=0.6, hypothesis_reasoning="r",
        code_alignment_score=0.8, code_reasoning="r",
        results_score=0.5, results_reasoning="r",
        style_categories=["trend_following"], insight="t",
    )
    rec.style = StyleVector.single(task["island"] if task["island"] < 8 else 0)
    return rec


@pytest.mark.skipif(not hasattr(multiprocessing, "get_context"),
                    reason="no multiprocessing contexts")
def test_run_dispatches_all_islands_and_places_every_candidate(tmp_path):
    import src.evolution.orchestrator as orchestrator

    orig = orchestrator.produce_candidate
    orchestrator.produce_candidate = _fake_producer

    class CheapFinish(PathARunner):
        def finish(self):
            return {"track": self.track.name, "path": self.path,
                    "occupied_cells": self.fm.cell_count()}

    try:
        cfg = load_config()
        track = cfg.tracks[0]
        runner = CheapFinish(cfg, track, "a", tier_specs()["tier1"])
        runner.run_dir = tmp_path / "tier1"
        runner.run_dir.mkdir(parents=True, exist_ok=True)
        runner.strategies_dir = tmp_path / "strategies"
        summary = runner.run(mp_context=multiprocessing.get_context("fork"))

        assert len(runner.records) == 9 + 9  # 9 seeds + 9 generated
        for island in range(9):
            sids = [e.entry_id for e in runner.fm.log if e.island == island]
            assert any(s.startswith("eqa") and s.endswith("g1c1") for s in sids), island
        assert summary["occupied_cells"] >= 9
    finally:
        orchestrator.produce_candidate = orig


@pytest.mark.skipif(not hasattr(multiprocessing, "get_context"),
                    reason="no multiprocessing contexts")
def test_run_multigeneration_migration_and_curation(tmp_path):
    """Tier 2/3-style run: >1 generation with migration and curation enabled
    (Path B, which has real _migrate / environmental selection). Verifies the
    concurrent run() loop drives migration + curation without error."""
    import re

    import src.evolution.orchestrator as orchestrator
    from src.evolution.orchestrator import TierSpec
    from src.evolution.path_b_run import PathBRunner

    orig = orchestrator.produce_candidate
    orchestrator.produce_candidate = _fake_producer

    class CheapFinish(PathBRunner):
        def finish(self):
            return {"track": self.track.name, "path": self.path,
                    "n_migration_events": len(self.migration_events)}

    try:
        cfg = load_config()
        track = cfg.tracks[0]
        tier = TierSpec("tier_smoke", generations=2, candidates_per_gen=1,
                        migration_interval=2, curation_interval=2)
        runner = CheapFinish(cfg, track, "b", tier)
        runner.run_dir = tmp_path / "tier_smoke"
        runner.run_dir.mkdir(parents=True, exist_ok=True)
        runner.strategies_dir = tmp_path / "strategies"
        for island in runner.islands.values():
            island.insights.path = tmp_path / f"tier_smoke/island_{island.island}" / "insights.csv"
        summary = runner.run(mp_context=multiprocessing.get_context("fork"))

        generated = [r for r in runner.records.values()
                     if re.match(r"eqb\d+g\d+c\d+$", r.strategy_id)]
        assert len(generated) == 9 * 2  # 9 islands x 2 generations
        assert summary["n_migration_events"] >= 1
        assert len(runner.migration_events) >= 1
        for island in range(9):
            assert runner.islands[island].population, island
        # curation pass ran at gen 2 without error (insight files exist)
        for island in range(9):
            assert (tmp_path / "tier_smoke" / f"island_{island}" / "insights.csv").exists()
    finally:
        orchestrator.produce_candidate = orig
        empty = Path(__file__).resolve().parents[1] / "outputs" / "evolution" / "equities" / "b" / "tier_smoke"
        if empty.exists():
            empty.rmdir()


@pytest.mark.skipif(not hasattr(multiprocessing, "get_context"),
                    reason="no multiprocessing contexts")
def test_run_path_a_multigeneration_migration(tmp_path):
    """Path A multi-generation run: migration copies archive entries between
    islands (Tier 2/3 exercise it), all islands keep archive coverage."""
    import re

    import src.evolution.orchestrator as orchestrator
    from src.evolution.orchestrator import TierSpec

    orig = orchestrator.produce_candidate
    orchestrator.produce_candidate = _fake_producer

    class CheapFinish(PathARunner):
        def finish(self):
            return {"track": self.track.name, "path": self.path,
                    "n_migration_events": len(self.migration_events),
                    "occupied_cells": self.fm.cell_count()}

    try:
        cfg = load_config()
        track = cfg.tracks[0]
        tier = TierSpec("tier_smoke", generations=2, candidates_per_gen=1,
                        migration_interval=2, curation_interval=0)
        runner = CheapFinish(cfg, track, "a", tier)
        runner.run_dir = tmp_path / "tier_smoke_a"
        runner.run_dir.mkdir(parents=True, exist_ok=True)
        runner.strategies_dir = tmp_path / "strategies"
        for island in runner.islands.values():
            island.insights.path = tmp_path / f"tier_smoke_a/island_{island.island}" / "insights.csv"
        summary = runner.run(mp_context=multiprocessing.get_context("fork"))

        generated = [r for r in runner.records.values()
                     if re.match(r"eqa\d+g\d+c\d+$", r.strategy_id)]
        assert len(generated) == 9 * 2
        assert summary["n_migration_events"] >= 1
        assert len(runner.migration_events) >= 1
        assert summary["occupied_cells"] >= 9
        for island in range(9):
            sids = [e.entry_id for e in runner.fm.log if e.island == island]
            assert any(s.endswith("g2c1") or s.endswith("g1c1") for s in sids), island
    finally:
        orchestrator.produce_candidate = orig
        empty = Path(__file__).resolve().parents[1] / "outputs" / "evolution" / "equities" / "a" / "tier_smoke"
        if empty.exists():
            empty.rmdir()
