"""Unit tests for extracted orchestrator components (BacktestRunner, CandidateFactory, load_seed_records)."""

import pytest
import pandas as pd
import numpy as np

from src.config.settings import load_config
from src.coders.team import BacktestResult
from src.evolution.orchestrator import (
    BacktestRunner,
    CandidateFactory,
    CandidateFactoryContext,
    _Island,
    load_seed_records,
)
from src.evolution.record import StrategyRecord
from src.roles.insights import InsightLog
from src.roles.schemas import Evaluation, Hypothesis


def test_load_seed_records_standalone():
    cfg = load_config()
    track = cfg.track("equities")
    records = load_seed_records(track, "a")
    assert isinstance(records, list)
    if records:
        assert records[0].track == "equities"
        assert records[0].path == "a"


def test_backtest_runner_standalone_initialization():
    cfg = load_config()
    track = cfg.track("equities")
    br = BacktestRunner(track)
    assert hasattr(br, "bench_train")
    assert hasattr(br, "bench_val")
    assert hasattr(br, "bench_test")
    assert "trading_days_per_month" in br.extra


# ---------------------------------------------------------------------------
# CandidateFactory standalone construction (Finding 2)
# ---------------------------------------------------------------------------

class _StubResearch:
    def generate_hypothesis(self, parent, cousins, insights, track):
        return Hypothesis("h", "r", "o", "e", "rl", "n")


class _StubCoder:
    def generate(self, sid, hypothesis, parent, cousins, track):
        result = BacktestResult(
            ok=True,
            metrics={
                "sharpe": 0.5, "sortino": 0.6, "ir": 0.3, "mdd": -0.2,
                "total_return": 0.4, "turnover": 0.1, "trading_frequency": 50.0,
                "n_fills": 120, "n_days": 100, "final_equity": 1.0e6,
            },
            info={},
        )
        return "def initialize(context):\n    pass\n", result


class _StubEvaluator:
    def evaluate(self, record, backtest_log, baseline_summary):
        return Evaluation(
            hypothesis_score=0.6, hypothesis_reasoning="r",
            code_alignment_score=0.8, code_reasoning="r",
            results_score=0.5, results_reasoning="r",
            style_categories=["trend_following"], insight="keep testing",
        )


def _bare_record(strategy_id: str, track_name: str, path: str, island: int) -> StrategyRecord:
    """A minimal StrategyRecord built directly, with no runner/seed loading
    involved -- stands in for a parent/cousin CandidateFactory can sample."""
    return StrategyRecord(
        strategy_id=strategy_id, track=track_name, path=path,
        island=island, generation=0,
        hypothesis=Hypothesis("h", "r", "o", "e", "rl", "n"),
        code="def initialize(context):\n    pass\n",
        metrics={
            "sharpe": 0.4, "sortino": 0.5, "ir": 0.2, "mdd": -0.1,
            "total_return": 0.1, "turnover": 0.05, "trading_frequency": 10.0,
            "n_fills": 5, "n_days": 100,
        },
        status="ok",
    )


def test_candidate_factory_constructs_and_generates_without_a_base_runner(tmp_path):
    """Finding 2's acceptance bar: CandidateFactory must be constructible and
    able to generate a real candidate given only the small, explicit pieces of
    state/collaborators it actually needs -- a records dict, an island map, a
    sampling function, and registration callbacks -- never a full BaseRunner
    (no BaseRunner/PathARunner/PathBRunner is constructed anywhere here)."""
    cfg = load_config()
    track = cfg.track("equities")

    parent = _bare_record("standalone_parent", track.name, "a", 0)
    cousin = _bare_record("standalone_cousin", track.name, "a", 0)
    records = {parent.strategy_id: parent, cousin.strategy_id: cousin}
    islands = {0: _Island(0, InsightLog(tmp_path / "insights.csv", k=50))}

    registered: list[StrategyRecord] = []
    logged: list[StrategyRecord] = []
    accepted: list[StrategyRecord] = []

    def sample(island: int):
        return parent, [cousin]

    def register(rec: StrategyRecord, code: str) -> None:
        records[rec.strategy_id] = rec
        registered.append(rec)

    context = CandidateFactoryContext(
        path="a",
        track=track,
        models=cfg.models,
        records=records,
        islands=islands,
        run_dir=tmp_path,
        strategies_dir=tmp_path / "strategies",
        sample=sample,
        register=register,
        log_record=logged.append,
        accept=accepted.append,
    )
    factory = CandidateFactory(
        context, BacktestRunner(track), _StubResearch(), _StubCoder(), _StubEvaluator(),
    )

    rec = factory.generate_candidate(island=0, generation=1, candidate=1)

    assert rec is not None
    assert rec.status == "ok"
    assert rec.strategy_id == "eqa0g1c1"
    assert rec.metrics["ir"] == 0.3
    # the callbacks CandidateFactory was given (not a BaseRunner) actually fired
    assert registered == [rec]
    assert logged == [rec]
    assert accepted == [rec]
    assert records[rec.strategy_id] is rec
