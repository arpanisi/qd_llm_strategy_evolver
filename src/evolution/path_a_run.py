"""Path A run engine: feature-map archive + Step 4A sampling + Step 8A/9A.

Shared across the track's 9 islands: one FeatureMap (with its flat Archive
Log) is the population structure; islands compete for cells but keep their own
tagged entries, insights, and migration behavior.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config.settings import RunConfig, TrackConfig
from src.engine.metrics import StrategyMetrics
from src.evolution.orchestrator import BaseRunner, TierSpec
from src.evolution.persistence import load_feature_map
from src.evolution.rng import island_rng, migration_rng
from src.evolution.worker import _copy_migrated_returns
from src.path_a.archive import ArchiveEntry
from src.path_a.sampling import PathASampler
from src.path_a.scoring import breadth_from_cells, rank_by_score

log = logging.getLogger("evolution.path_a")


class PathARunner(BaseRunner):
    def _init_from_seeds(self) -> None:
        self.fm = load_feature_map(
            self.track.name,
            self.cfg.path_a.ir_near_tie,
            self.cfg.path_a.turnover_tiebreak,
        )
        self.sampler = PathASampler(self.fm, self.cfg.path_a, island_rng(
            self.cfg.seed, self.track.name, "a", 0))
        self.records_rows: list[dict] = []
        self.migration_events: list[dict] = []
        self.val_scores: dict[str, float] = {}
        self.test_metrics: dict[str, dict] = {}
        for rec in self.records.values():
            self.val_scores[rec.strategy_id] = float("nan")

    def _sample(self, island: int):
        parent = self.sampler.sample_parent(island)
        cousins = self.sampler.sample_cousins(island, parent)
        return parent, cousins

    def _log_record(self, rec) -> None:
        self.records_rows.append(rec.to_archive_row())

    def _entry_from_record(self, rec) -> ArchiveEntry:
        return ArchiveEntry(
            entry_id=rec.strategy_id,
            island=rec.island,
            generation=rec.generation,
            strategy_name=rec.strategy_id,
            style=rec.style,
            trading_frequency=rec.metric("trading_frequency"),
            max_drawdown=rec.metric("mdd"),
            sharpe=rec.metric("sharpe"),
            sortino=rec.metric("sortino"),
            total_return=rec.metric("total_return"),
            turnover=rec.metric("turnover"),
            information_ratio=rec.metric("ir"),
            lineage_parents=rec.parents,
            lineage_cousins=rec.cousins,
        )

    def _accept(self, rec) -> None:
        if rec.is_failed or rec.style is None:
            return
        entry = self._entry_from_record(rec)
        self.fm.place(entry)
        self.codes[entry.entry_id] = rec.code

    # -- migration (Step 3) -------------------------------------------------

    def _migrate(self, generation: int) -> None:
        rng = migration_rng(self.cfg.seed, self.track.name, "a")
        n = self.cfg.evolution.num_islands
        for src in sorted(self.islands):
            # Only ok records ever reach the feature map (`_accept` guards on
            # rec.is_failed), so every archive-log entry is migratable.
            entries = [e for e in self.fm.log if e.island == src]
            if not entries:
                continue
            ranked = rank_by_score(entries, self.fm.ir_near_tie, self.fm.turnover_tiebreak)
            k = max(1, int(round(0.10 * len(ranked))))
            dst = (src + (1 if rng.random() < 0.5 else -1)) % n
            for winner in ranked[:k]:
                mid = f"{winner.strategy_name}:mig{src}->{dst}:g{generation}"
                copy = _copy_entry(winner, mid, dst, generation)
                self.fm.place(copy)
                self.codes[copy.entry_id] = self.codes.get(winner.entry_id, "")
                src_rec = self.records.get(winner.entry_id)
                if src_rec is not None:
                    import copy as _copy

                    mig_rec = _copy.copy(src_rec)
                    mig_rec.strategy_id = mid
                    mig_rec.island = dst
                    mig_rec.generation = generation
                    self.records[mid] = mig_rec
                _copy_migrated_returns(self.run_dir, winner.entry_id, mid)
                self.migration_events.append({
                    "generation": generation, "source_island": src,
                    "destination_island": dst, "entry_id": copy.entry_id,
                    "source_entry_id": winner.entry_id,
                })
        log.info("[%s Path A] migration event at generation %d", self.track.name, generation)

    def _persist_generation(self, generation: int) -> None:
        self.fm.log.to_dataframe().to_csv(self.run_dir / "archive_log.csv", index=False)
        self.fm.cells_dataframe().to_csv(self.run_dir / "archive_cells.csv", index=False)
        pd.DataFrame(self.records_rows).to_csv(self.run_dir / "records.csv", index=False)
        if self.migration_events:
            pd.DataFrame(self.migration_events).to_csv(
                self.run_dir / "migration_events.csv", index=False)

    # -- final selection discipline (Step 8A) --------------------------------

    def _recompute_validation_scores(self) -> None:
        """Re-run each feature-map occupant's frozen code on the validation
        window; the result is that entry's 'official' Combined Score."""
        seen: set[str] = set()
        for entry in self.fm.cells.values():
            if entry.entry_id in seen:
                continue
            seen.add(entry.entry_id)
            code = self.codes.get(entry.entry_id, "")
            if not code:
                self.val_scores[entry.entry_id] = float("nan")
                continue
            try:
                m = self._rerun(code, str(self.track.validation.start),
                                str(self.track.validation.end), self.bench_val)
                self.val_scores[entry.entry_id] = float(m.ir)
            except Exception as e:  # noqa: BLE001
                log.warning("validation re-run failed for %s: %s", entry.entry_id, e)
                self.val_scores[entry.entry_id] = float("nan")

    def _best_per_island(self) -> dict[int, str]:
        """Highest validation-window Combined Score among occupant entries
        tagged with each island (Step 8A 'single best strategy per island')."""
        best: dict[int, str] = {}
        for entry in self.fm.cells.values():
            score = self.val_scores.get(entry.entry_id, float("nan"))
            if np.isnan(score):
                continue
            if entry.island not in best or score > self.val_scores[best[entry.island]]:
                best[entry.island] = entry.entry_id
        return best

    def _rerun_test(self, sid: str, code: str) -> StrategyMetrics:
        """Test-window re-run of frozen code that also persists the test-window
        daily returns. Step 8A only ever test-reruns the per-island-best
        strategies (one per island, picked by validation-window Combined
        Score), and Step 13's DSR target is one of them, so its test returns
        are guaranteed to be persisted here — no additional backtest."""
        import numpy as np
        from src.engine.runtime import run_strategy

        perf, m, _ = run_strategy(
            code, self.track,
            str(self.track.test.start), str(self.track.test.end),
            self.track.starting_cash,
            benchmark_returns=self.bench_test, extra=self.extra,
        )
        test_returns_dir = self.run_dir / "test_returns"
        test_returns_dir.mkdir(parents=True, exist_ok=True)
        rets = perf["returns"].astype(float).fillna(0.0)
        np.savetxt(test_returns_dir / f"{sid}.csv", np.asarray(rets, dtype=float))
        return m

    def finish(self) -> dict:
        self._recompute_validation_scores()
        best = self._best_per_island()
        for island, entry_id in best.items():
            code = self.codes.get(entry_id, "")
            if not code:
                continue
            m = self._rerun_test(entry_id, code)
            self.test_metrics[entry_id] = metrics_dict_ok(m)

        breadth = breadth_from_cells(self.fm)
        summary = {
            "track": self.track.name,
            "path": "a",
            "tier": self.tier.name,
            "breadth": breadth,
            "best_per_island": {
                str(island): {
                    "entry_id": entry_id,
                    "validation_ir": self.val_scores[entry_id],
                    "test": self.test_metrics.get(entry_id, {}),
                }
                for island, entry_id in sorted(best.items())
            },
            "dsr_target": _select_dsr_target(best, self.val_scores),
            "occupied_cells": self.fm.cell_count(),
            "archive_log_entries": len(self.fm.log),
            "n_migration_events": len(self.migration_events),
        }
        with open(self.run_dir / "summary.json", "w") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
        return summary


def metrics_dict_ok(m) -> dict:
    return {
        "sharpe": m.sharpe, "sortino": m.sortino, "ir": m.ir, "mdd": m.mdd,
        "total_return": m.total_return, "turnover": m.turnover,
        "trading_frequency": m.trading_frequency, "n_fills": m.n_fills,
    }


def _select_dsr_target(best: dict, val_scores: dict) -> Optional[str]:
    """Step 13 DSR target: among the per-island-best strategies that Step 8A
    already test-reruns (best: island -> entry_id), the one with the highest
    validation-window Combined Score (validation IR). Only strategies whose
    validation re-run produced a finite score are eligible; ties resolve to
    the lowest island number. None if no island has an eligible best."""
    finite = [
        (island, sid) for island, sid in sorted(best.items())
        if np.isfinite(val_scores.get(sid, float("nan")))
    ]
    if not finite:
        return None
    return max(finite, key=lambda is_sid: (val_scores[is_sid[1]], -is_sid[0]))[1]


def _copy_entry(entry: ArchiveEntry, new_id: str, dst_island: int,
                generation: int) -> ArchiveEntry:
    import dataclasses

    copy = dataclasses.replace(
        entry,
        entry_id=new_id,
        island=dst_island,
        generation=generation,
        placed=False,
        placed_reason="",
    )
    return copy
