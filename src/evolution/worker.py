"""Candidate worker: the Steps 5/6/7 pipeline as a pure, picklable function.

One process-level worker per island (the plan's concurrency model: islands
run concurrently, candidate generation stays sequential *within* an island).
``produce_candidate`` is deliberately a free function operating only on the
``task`` dict it is given: it never touches a runner, feature map, population,
reference table, or any other shared evolution state. The main process owns
all sampling (before the call) and all placement/selection (after the call),
so archive-cell and reference-point writes can never race.

Everything crossing the process boundary (track/models configs, parent and
cousin records, insight lines, benchmark series, and the returned StrategyRecord)
is pickleable; the worker rebuilds its own LLM client + roles per call.
"""

from __future__ import annotations

import traceback
from pathlib import Path

from src.coders.team import BacktestResult, CoderTeam
from src.config.settings import ModelConfig, TrackConfig
from src.data.taxonomy import StyleVector
from src.evolution.record import StrategyRecord
from src.llm.client import OpenRouterClient
from src.path_b.population import objectives_from_metrics
from src.roles.evaluate import EvaluationRole
from src.roles.research import ResearchRole
from src.roles.schemas import Hypothesis


def _metrics_payload(m) -> dict[str, float]:
    return {
        "sharpe": m.sharpe,
        "sortino": m.sortino,
        "ir": m.ir,
        "mdd": m.mdd,
        "total_return": m.total_return,
        "turnover": m.turnover,
        "trading_frequency": m.trading_frequency,
        "n_fills": m.n_fills,
        "n_days": m.n_days,
        "final_equity": m.final_equity,
    }


def _persist_returns(returns_dir, sid: str, returns) -> None:
    """Write a strategy's window daily-return series (one float per line).
    Used by Step 13's PBO/DSR, which must be computed from already-persisted
    returns with no re-backtest."""
    import numpy as np

    d = Path(returns_dir)
    d.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(list(returns), dtype=float)
    np.savetxt(d / f"{sid}.csv", arr)


def _copy_migrated_returns(run_dir, src_sid: str, dst_sid: str) -> None:
    """A migrated strategy is a copy with identical performance, so its
    returns file is copied under the new strategy id (Step 13's PBO reads the
    final population's returns, which may include migrated copies)."""
    import shutil

    src = Path(run_dir) / "returns" / f"{src_sid}.csv"
    if not src.exists():
        return
    dst_dir = Path(run_dir) / "returns"
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst_dir / f"{dst_sid}.csv")


def run_backtest(code: str, track: TrackConfig, bench, extra: dict) -> BacktestResult:
    """Worker-side train-window backtest. Never raises; failures become a
    BacktestResult with the traceback preserved (Step 6 stage-3 semantics)."""
    from src.engine.runtime import run_strategy

    try:
        perf, m, info = run_strategy(
            code, track,
            str(track.train.start), str(track.train.end),
            track.starting_cash,
            benchmark_returns=bench, extra=extra,
        )
        info["returns"] = list(perf["returns"].astype(float).fillna(0.0))
        return BacktestResult(ok=True, metrics=_metrics_payload(m), info=info)
    except Exception as e:  # noqa: BLE001
        # format_exc() can itself crash on Python 3.12 for frames compiled via
        # exec()'d strategy code; fall back to type+message so the real error
        # is never masked.
        try:
            err = traceback.format_exc()
        except Exception:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        return BacktestResult(ok=False, metrics={}, info={}, error=err)


def _failed(task: dict, sid: str, error: str) -> StrategyRecord:
    rec = StrategyRecord(
        strategy_id=sid,
        track=task["track"].name,
        path=task["path"],
        island=task["island"],
        generation=task["generation"],
        hypothesis=Hypothesis("", "", "", "", "", ""),
        code="",
        status="failed",
        failure_error=str(error)[:2000],
    )
    return rec


def produce_candidate_core(
    task: dict,
    research: ResearchRole,
    coder: CoderTeam,
    evaluator: EvaluationRole,
) -> StrategyRecord:
    """Steps 5/6/7 against injected roles. Pure: reads only the task dict
    and the parent/cousin records inside it; writes nothing shared."""
    sid = task["sid"]
    parent = task["parent"]
    cousins = task["cousins"]

    try:
        hypothesis = research.generate_hypothesis(parent, cousins, task["insights"], task["track"])
    except Exception as e:  # noqa: BLE001
        return _failed(task, sid, f"Step 5 research failed: {e}")

    try:
        code, result = coder.generate(sid, hypothesis, parent, cousins, task["track"])
    except Exception as e:  # noqa: BLE001
        return _failed(task, sid, f"Step 6 coder failed: {e}")

    rec = StrategyRecord(
        strategy_id=sid,
        track=task["track"].name,
        path=task["path"],
        island=task["island"],
        generation=task["generation"],
        hypothesis=hypothesis,
        code=code,
        parents=[parent.strategy_id],
        cousins=[c.strategy_id for c in cousins],
    )
    if not result.ok:
        rec.status = "failed"
        rec.failure_error = (result.error or "")[:2000]
        return rec

    rec.metrics = result.metrics
    rec.combined_score = rec.metric("ir")
    rec.objectives = objectives_from_metrics(
        rec.metric("sharpe"), rec.metric("sortino"),
        rec.metric("total_return"), rec.metric("mdd"),
    )

    rets = result.info.get("returns")
    returns_dir = task.get("returns_dir")
    if rets and returns_dir:
        _persist_returns(returns_dir, sid, rets)

    backtest_log = (
        f"n_rejections={result.info.get('n_rejections', 0)}; "
        f"n_fills={int(rec.metric('n_fills'))}; n_days={int(rec.metric('n_days'))}"
    )
    try:
        evaluation = evaluator.evaluate(rec, backtest_log, task["baseline_summary"])
    except Exception as e:  # noqa: BLE001
        rec.status = "failed"
        rec.failure_error = f"Step 7 evaluation failed: {e}"
        return rec

    rec.evaluation = evaluation
    rec.style = StyleVector.from_names(evaluation.style_categories)
    return rec


def produce_candidate(task: dict) -> StrategyRecord:
    """Worker entry point: build roles from the task's config and run Steps 5/6/7."""
    models: ModelConfig = task["models"]
    track: TrackConfig = task["track"]
    client = OpenRouterClient(
        models.openrouter_base_url,
        max_retries=models.max_retries,
        timeout_s=models.request_timeout_s,
    )
    research = ResearchRole(client, models)
    coder = CoderTeam(
        client, models,
        lambda code: run_backtest(code, track, task["bench_train"], task["extra"]),
    )
    evaluator = EvaluationRole(client, models)
    return produce_candidate_core(task, research, coder, evaluator)
