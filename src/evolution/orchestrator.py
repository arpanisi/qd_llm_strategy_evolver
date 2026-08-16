"""The per-track / per-path evolution engine (Steps 3, 5, 6, 7, 8A, 8B).

One runner per (track, path) pair, driven by a TierSpec. Every candidate on an
island flows through Step 5 (hypothesis) -> Step 6 (implement -> backtest ->
refine) -> Step 7 (evaluate) -> Step 8A/8B (archive placement / population
environmental selection). Migration (Step 3) and insight curation (Step 7)
fire at their configured intervals.

Concurrency follows the locked Runtime Plan: the 9 islands run concurrently
with each other, while candidate generation stays strictly sequential within
each island. ``produce_candidate`` (src.evolution.worker) runs Steps 5/6/7 in
worker processes as a pure function of already-sampled inputs; this runner
samples parents/cousins in the main process, dispatches one worker per island
per candidate slot, then does all archive/population/reference-table writes
itself on the single main thread. No locking scheme is needed because no
shared-state write ever happens concurrently.

Train/validation/test discipline is structural: evolution only ever runs over
the training window; validation re-runs of frozen code select finals; test is
touched exactly once, at the very end, through a separate method that nothing
during evolution can reach.
"""

from __future__ import annotations

import json
import logging
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.coders.team import BacktestResult, CoderTeam, write_strategy_code
from src.config.settings import ModelConfig, RunConfig, TrackConfig
from src.data.seeds import load_seeds, trading_days_per_month
from src.data.taxonomy import StyleVector
from src.engine.metrics import StrategyMetrics
from src.engine.runtime import run_strategy
from src.evolution.persistence import (
    load_feature_map,
    load_ideal_point,
    load_references,
)
from src.evolution.record import StrategyRecord, format_strategy_history
from src.evolution.rng import island_rng, migration_rng
from src.evolution.worker import produce_candidate, produce_candidate_core
from src.llm.client import OpenRouterClient
from src.path_a.archive import ArchiveEntry
from src.path_a.sampling import PathASampler
from src.path_a.scoring import better, best_per_island, breadth_from_cells, rank_by_score
from src.path_b.mating import PathBMating
from src.path_b.population import (
    ReferencePointTable,
    nadir_from_front,
    normalize,
    objectives_from_metrics,
)
from src.path_b.selection import (
    breadth_from_population,
    environmental_selection,
    front_1_indices,
    non_dominated_sort,
)
from src.roles.evaluate import EvaluationRole
from src.roles.insights import InsightLog
from src.roles.research import ResearchRole

log = logging.getLogger("evolution.orchestrator")

OUTPUTS = Path(__file__).resolve().parents[2] / "outputs" / "evolution"


@dataclass(frozen=True)
class TierSpec:
    name: str
    generations: int
    candidates_per_gen: int
    migration_interval: int = 0      # 0 = disabled
    curation_interval: int = 0       # 0 = disabled


def tier_specs() -> dict[str, TierSpec]:
    return {
        "tier1": TierSpec("tier1", 1, 1),
        "tier2": TierSpec("tier2", 8, 2, migration_interval=4, curation_interval=8),
        "tier3": TierSpec("tier3", 150, 8, migration_interval=10, curation_interval=50),
    }


def _track_abbr(track: TrackConfig) -> str:
    return "fu" if track.is_futures else "eq"


def metrics_dict(m: StrategyMetrics) -> dict[str, float]:
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


class _Island:
    def __init__(self, island: int, insights: InsightLog) -> None:
        self.island = island
        self.insights = insights
        self.incoming: list[StrategyRecord] = []  # Path B: migrated this gen
        # Path B current-population bookkeeping, refreshed at selection time
        self.population: list[StrategyRecord] = []
        self.norm_objectives: Optional[np.ndarray] = None
        self.ref_table = None
        self.migration_log: list[dict] = []


# ---------------------------------------------------------------------------
# Extracted Helper 1: Backtest Execution Component
# ---------------------------------------------------------------------------

class BacktestRunner:
    """Manages backtest execution, benchmark returns loading, and metrics derivation."""

    def __init__(self, track: TrackConfig) -> None:
        self.track = track
        self.bench_train = self._rb_returns("train")
        self.bench_val = self._rb_returns("validation")
        self.bench_test = self._rb_returns("test")
        self.extra = {"trading_days_per_month": trading_days_per_month(track)}
        self.baseline_summary = self._baseline_summary()

    def _rb_returns(self, window: str) -> pd.Series:
        p = Path(__file__).resolve().parents[2] / "outputs" / "baselines" / self.track.name / f"rb_returns_{window}.csv"
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        return df.iloc[:, 0]

    def _baseline_summary(self) -> str:
        r = self.bench_train
        return (f"daily mean={r.mean():.6f}, std={r.std():.6f}, "
                f"annualized Sharpe={r.mean()/r.std(ddof=0)*np.sqrt(self.track.bars_per_year):.3f}" if r.std() else "flat")

    def train_backtest(self, code: str) -> BacktestResult:
        try:
            perf, m, info = run_strategy(
                code, self.track,
                str(self.track.train.start), str(self.track.train.end),
                self.track.starting_cash,
                benchmark_returns=self.bench_train, extra=self.extra,
            )
            info["returns"] = list(perf["returns"].astype(float).fillna(0.0))
            return BacktestResult(ok=True, metrics=metrics_dict(m), info=info)
        except Exception as e:  # noqa: BLE001
            try:
                err = traceback.format_exc()
            except Exception:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
            return BacktestResult(ok=False, metrics={}, info={}, error=err)

    def rerun(self, code: str, start: str, end: str, bench: pd.Series) -> StrategyMetrics:
        _, m, _ = run_strategy(
            code, self.track, start, end, self.track.starting_cash,
            benchmark_returns=bench, extra=self.extra,
        )
        return m


# ---------------------------------------------------------------------------
# Extracted Helper 2: Seed Initialization Component
# ---------------------------------------------------------------------------

def _docstring(code: str) -> str:
    for line in code.splitlines():
        line = line.strip()
        if line.startswith('"""'):
            return line.strip('"').strip()
        if line.startswith("#"):
            continue
    return "seed strategy"


class _SeedHypothesis:
    """Minimal hypothesis wrapper for seed strategies (their module docstring)."""

    def __init__(self, doc: str) -> None:
        self.hypothesis = doc
        self.rationale = doc
        self.objectives = ""
        self.expected_insights = ""
        self.risks_limitations = ""
        self.next_step_ideas = ""


class _EmptyHypothesis:
    def __init__(self) -> None:
        self.hypothesis = ""
        self.rationale = ""
        self.objectives = ""
        self.expected_insights = ""
        self.risks_limitations = ""
        self.next_step_ideas = ""


def load_seed_records(track: TrackConfig, path: str) -> list[StrategyRecord]:
    """Seed initialization helper loading baseline seed records for a track/path."""
    seeds = load_seeds(track)
    seeds_file = Path(__file__).resolve().parents[2] / "outputs" / "seeds" / track.name / "seeds.csv"
    if not seeds_file.exists():
        return []
    seeds_df = pd.read_csv(seeds_file)
    records = []
    for _, row in seeds_df.iterrows():
        stem = str(row["seed"])
        island = int(stem.split("_")[1])
        doc = _docstring(seeds.get(stem, ""))
        rec = StrategyRecord(
            strategy_id=f"{stem}:g0",
            track=track.name,
            path=path,
            island=island,
            generation=0,
            hypothesis=_SeedHypothesis(doc),
            code=seeds.get(stem, ""),
            metrics={
                "sharpe": float(row["Sharpe Ratio"]),
                "sortino": float(row["Sortino Ratio"]),
                "ir": float(row["Information Ratio"]),
                "mdd": float(row["Max Drawdown"]),
                "total_return": float(row["Total Return"]),
                "turnover": float(row["Turnover"]),
                "trading_frequency": float(row["Trading Frequency"]),
                "n_fills": int(row["n_fills"]),
                "n_days": int(row["n_days"]),
            },
            status="ok",
        )
        rec.style = StyleVector.single(island) if island < 8 else StyleVector(*(False,) * 8)
        rec.objectives = objectives_from_metrics(rec.metric("sharpe"), rec.metric("sortino"),
                                                 rec.metric("total_return"), rec.metric("mdd"))
        rec.combined_score = rec.metric("ir")
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Extracted Helper 3: Candidate Generation & Task Factory
# ---------------------------------------------------------------------------

@dataclass
class CandidateFactoryContext:
    """The specific state and collaborators CandidateFactory needs to build,
    register, and log a candidate -- decoupled from the rest of BaseRunner.

    This is what makes CandidateFactory constructible and testable on its own:
    a caller passes exactly these pieces (a records store, an island map, a
    sampling function, a registration callback, etc.) instead of a full,
    already-running BaseRunner instance. ``records`` and ``islands`` are
    shared, mutable references (the same dict objects the owning runner reads
    and writes), not copies -- candidates registered through ``register`` or
    ``accept`` are visible to whatever else holds that dict.
    """

    path: str
    track: TrackConfig
    models: ModelConfig
    records: dict[str, StrategyRecord]
    islands: dict[int, "_Island"]
    run_dir: Path
    strategies_dir: Path
    sample: Callable[[int], tuple[object, list[object]]]
    register: Callable[[StrategyRecord, str], None]
    log_record: Callable[[StrategyRecord], None]
    accept: Callable[[StrategyRecord], None]


class CandidateFactory:
    """Manages candidate strategy creation, task inputs packaging, and failure logging."""

    def __init__(
        self,
        context: CandidateFactoryContext,
        backtest_runner: BacktestRunner,
        research: ResearchRole,
        coder: CoderTeam,
        evaluator: EvaluationRole,
    ) -> None:
        self.context = context
        self.backtest_runner = backtest_runner
        self.research = research
        self.coder = coder
        self.evaluator = evaluator

    def strategy_id(self, island: int, generation: int, candidate: int) -> str:
        return f"{_track_abbr(self.context.track)}{self.context.path}{island}g{generation}c{candidate}"

    def task_inputs(self, island: int, generation: int, candidate: int) -> dict:
        sid = self.strategy_id(island, generation, candidate)
        parent, cousins = self.context.sample(island)
        parent_rec = self.context.records[parent.entry_id if hasattr(parent, "entry_id") else parent.strategy_id]
        cousin_recs = [
            self.context.records[c.entry_id if hasattr(c, "entry_id") else c.strategy_id]
            for c in cousins
        ]
        return {
            "sid": sid,
            "island": island,
            "generation": generation,
            "path": self.context.path,
            "track": self.context.track,
            "models": self.context.models,
            "parent": parent_rec,
            "cousins": cousin_recs,
            "insights": self.context.islands[island].insights.recent(),
            "baseline_summary": self.backtest_runner.baseline_summary,
            "bench_train": self.backtest_runner.bench_train,
            "extra": self.backtest_runner.extra,
            "returns_dir": self.context.run_dir / "returns",
        }

    def failed_record(self, sid: str, island: int, generation: int, error: str) -> StrategyRecord:
        return StrategyRecord(
            strategy_id=sid, track=self.context.track.name, path=self.context.path,
            island=island, generation=generation,
            hypothesis=_EmptyHypothesis(), code="", status="failed",
            failure_error=str(error)[:2000],
        )

    def log_failed(self, sid: str, island: int, generation: int, error: str) -> StrategyRecord:
        log.warning("[%s %s] candidate %s failed (island %s, gen %d): %s",
                    self.context.track.name, self.context.path, sid, island, generation, str(error)[:500])
        rec = self.failed_record(sid, island, generation, error)
        self.context.register(rec, "")
        self.context.log_record(rec)
        return rec

    def handle_produced(self, rec: Optional[StrategyRecord], island: int, generation: int) -> None:
        if rec is None:
            return
        self.context.register(rec, rec.code)
        if rec.code:
            write_strategy_code(self.context.strategies_dir, rec.strategy_id, rec.code)
        self.context.log_record(rec)
        if rec.status != "ok":
            log.warning("[%s %s] candidate %s failed (island %s, gen %d): %s",
                        self.context.track.name, self.context.path, rec.strategy_id, island,
                        generation, (rec.failure_error or "")[:500])
            return
        self.context.islands[island].insights.add(
            rec.evaluation.insight, rec.evaluation.results_score, rec.strategy_id, generation
        )
        self.context.accept(rec)

    def generate_candidate(self, island: int, generation: int, candidate: int) -> Optional[StrategyRecord]:
        sid = self.strategy_id(island, generation, candidate)
        try:
            task = self.task_inputs(island, generation, candidate)
        except Exception as e:  # noqa: BLE001
            return self.log_failed(sid, island, generation, e)
        try:
            rec = produce_candidate_core(task, self.research, self.coder, self.evaluator)
        except Exception as e:  # noqa: BLE001
            rec = self.failed_record(sid, island, generation, f"produce failed: {e}")
        self.handle_produced(rec, island, generation)
        return rec


# ---------------------------------------------------------------------------
# BaseRunner: Evolutionary Orchestration Engine
# ---------------------------------------------------------------------------

class BaseRunner:
    """Base evolutionary orchestrator managing population loops and migration."""

    def __init__(self, cfg: RunConfig, track: TrackConfig, path: str,
                 tier: TierSpec) -> None:
        self.cfg = cfg
        self.track = track
        self.path = path
        self.tier = tier
        self.abbr = _track_abbr(track)
        self.run_dir = OUTPUTS / track.name / path / tier.name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.backtest_runner = BacktestRunner(track)

        self.client = OpenRouterClient(
            cfg.models.openrouter_base_url,
            max_retries=cfg.models.max_retries,
            timeout_s=cfg.models.request_timeout_s,
        )
        self.research = ResearchRole(self.client, cfg.models)
        self.coder = CoderTeam(self.client, cfg.models, self._train_backtest)
        self.evaluator = EvaluationRole(self.client, cfg.models)

        self.records: dict[str, StrategyRecord] = {}   # strategy_id -> record
        self.codes: dict[str, str] = {}                # strategy_id -> code
        self.strategies_dir = Path(__file__).resolve().parents[2] / "src" / "strategies" / track.name

        k = cfg.models.max_refinement_attempts
        self.islands = {
            i: _Island(i, InsightLog(self.run_dir / f"island_{i}" / "insights.csv", k=50))
            for i in range(cfg.evolution.num_islands)
        }

        # records/codes/islands must already exist: CandidateFactory holds direct
        # references to them, not a reference to self, so it is constructible
        # and testable given only this same kind of context (see Context above).
        self.candidate_factory = CandidateFactory(
            CandidateFactoryContext(
                path=self.path, track=self.track, models=cfg.models,
                records=self.records, islands=self.islands,
                run_dir=self.run_dir, strategies_dir=self.strategies_dir,
                sample=self._sample, register=self._register,
                log_record=self._log_record, accept=self._accept,
            ),
            self.backtest_runner, self.research, self.coder, self.evaluator,
        )
        self._load_seed_records()

    # -- Properties & Delegations for Backtest & Candidate components ------------

    @property
    def bench_train(self) -> pd.Series:
        return self.backtest_runner.bench_train

    @property
    def bench_val(self) -> pd.Series:
        return self.backtest_runner.bench_val

    @property
    def bench_test(self) -> pd.Series:
        return self.backtest_runner.bench_test

    @property
    def extra(self) -> dict:
        return self.backtest_runner.extra

    @property
    def baseline_summary(self) -> str:
        return self.backtest_runner.baseline_summary

    def _rb_returns(self, window: str) -> pd.Series:
        return self.backtest_runner._rb_returns(window)

    def _baseline_summary(self) -> str:
        return self.backtest_runner._baseline_summary()

    def _train_backtest(self, code: str) -> BacktestResult:
        return self.backtest_runner.train_backtest(code)

    def _rerun(self, code: str, start: str, end: str, bench: pd.Series) -> StrategyMetrics:
        return self.backtest_runner.rerun(code, start, end, bench)

    def _seed_records(self) -> list[StrategyRecord]:
        return load_seed_records(self.track, self.path)

    def _load_seed_records(self) -> None:
        for rec in self._seed_records():
            self.records[rec.strategy_id] = rec
            self.codes[rec.strategy_id] = rec.code
        self._init_from_seeds()

    def _init_from_seeds(self) -> None:
        raise NotImplementedError

    def _register(self, record: StrategyRecord, code: str) -> None:
        self.records[record.strategy_id] = record
        self.codes[record.strategy_id] = code

    # -- path hooks ----------------------------------------------------------

    def _sample(self, island: int) -> tuple[object, list[object]]:
        raise NotImplementedError

    def _log_record(self, rec: StrategyRecord) -> None:
        raise NotImplementedError

    def _accept(self, rec: StrategyRecord) -> None:
        raise NotImplementedError

    def _migrate(self, generation: int) -> None:
        raise NotImplementedError

    def _persist_generation(self, generation: int) -> None:
        raise NotImplementedError

    def run(self, mp_context=None) -> dict:
        self._init_populations()
        spec = self.tier
        n_workers = min(len(self.islands), max(1, os.cpu_count() or 4))
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as pool:
            for generation in range(1, spec.generations + 1):
                for candidate in range(1, spec.candidates_per_gen + 1):
                    in_flight: dict[int, tuple[str, object]] = {}
                    for island in sorted(self.islands):
                        sid = self.candidate_factory.strategy_id(island, generation, candidate)
                        try:
                            task = self.candidate_factory.task_inputs(island, generation, candidate)
                        except Exception as e:  # noqa: BLE001
                            self.candidate_factory.log_failed(sid, island, generation, e)
                            continue
                        in_flight[island] = (sid, pool.submit(produce_candidate, task))
                    for island in sorted(in_flight):
                        sid, future = in_flight[island]
                        try:
                            rec = future.result()
                        except Exception as e:  # noqa: BLE001
                            rec = self.candidate_factory.failed_record(sid, island, generation,
                                                      f"worker crashed: {e}")
                        self.candidate_factory.handle_produced(rec, island, generation)
                for island in sorted(self.islands):
                    self._after_candidates(island, generation)
                self._persist_generation(generation)
                if spec.migration_interval and generation % spec.migration_interval == 0:
                    self._migrate(generation)
                if spec.curation_interval and generation % spec.curation_interval == 0:
                    for isl in self.islands.values():
                        isl.insights.curate()
                log.info("[%s %s %s] gen %d done", self.track.name, self.path, spec.name, generation)
        return self.finish()

    def _after_candidates(self, island: int, generation: int) -> None:
        pass

    def _init_populations(self) -> None:
        pass

    def finish(self) -> dict:
        raise NotImplementedError
