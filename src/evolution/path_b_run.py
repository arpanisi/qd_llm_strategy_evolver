"""Path B run engine: per-island NSGA-III populations + Step 4B mating.

Each island keeps an independent population targeting P=56 individuals. Every
generation's offspring (plus migrated individuals) join the current population
into one pool, which is normalized against the running ideal/nadir and trimmed
back to P via Step 8B's front fill + reference-point niche selection.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src.config.settings import RunConfig, TrackConfig
from src.evolution.orchestrator import BaseRunner, TierSpec
from src.evolution.persistence import load_ideal_point, load_references
from src.evolution.rng import island_rng, migration_rng
from src.evolution.worker import _copy_migrated_returns
from src.path_b.mating import PathBMating
from src.path_b.population import (
    OBJECTIVE_NAMES,
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

log = logging.getLogger("evolution.path_b")


class PathBRunner(BaseRunner):
    def _init_from_seeds(self) -> None:
        self.refs = load_references(self.track.name)
        self.ideal = load_ideal_point(self.track.name)
        self.pop_target = self.cfg.path_b.pop_target
        self.pop_rows: list[dict] = []
        self.migration_events: list[dict] = []
        for i, island in self.islands.items():
            island.population = [
                rec for rec in self.records.values()
                if rec.island == i and rec.strategy_id.startswith("seed_")
            ]
            island.ref_table = ReferencePointTable(self.refs)
        for rec in self.records.values():
            if rec.strategy_id.startswith("seed_"):
                self.pop_rows.append(_pop_row(rec, accepted=True))
        self._refresh_all()

    def _refresh_island(self, island_id: int) -> None:
        island = self.islands[island_id]
        pool = island.population + island.incoming
        island.incoming = []
        if not pool:
            return
        obj = np.array([r.objectives for r in pool], dtype=float)
        fronts = non_dominated_sort(obj)
        nadir = nadir_from_front(obj[fronts[0]])
        norm = normalize(obj, self.ideal.z_star, nadir, self.cfg.path_b.nadir_floor)
        turnover = np.array([r.metric("turnover") for r in pool], dtype=float)
        sel = environmental_selection(norm, turnover, self.refs, self.pop_target,
                                      island_rng(self.cfg.seed, self.track.name, "b", island_id))
        island.population = [pool[i] for i in sel]
        ids = [r.strategy_id for r in island.population]
        island.ref_table.recompute(ids, np.array([norm[i] for i in sel]))
        island.norm_objectives = np.array([norm[i] for i in sel])

    def _refresh_all(self) -> None:
        for island_id in self.islands:
            self._refresh_island(island_id)

    def _sample(self, island_id: int):
        island = self.islands[island_id]
        mating = PathBMating(self.refs, island_rng(self.cfg.seed, self.track.name, "b", island_id))
        id_to_index = {r.strategy_id: i for i, r in enumerate(island.population)}
        parent, cousins = mating.sample_parent_and_cousins(
            island.population, island.norm_objectives, island.ref_table, id_to_index)
        return parent, cousins

    def _log_record(self, rec) -> None:
        row = rec.to_pop_row()
        row["accepted"] = False
        self.pop_rows.append(row)

    def _accept(self, rec) -> None:
        if rec.is_failed or rec.objectives is None:
            return
        self.ideal.observe([rec.objectives])
        self.islands[rec.island].incoming.append(rec)

    def _after_candidates(self, island: int, generation: int) -> None:
        self._refresh_island(island)
        self._flag_accepted(island)

    def _flag_accepted(self, island_id: int) -> None:
        current = {r.strategy_id for r in self.islands[island_id].population}
        for row in self.pop_rows:
            row["accepted"] = row["accepted"] or row["strategy_id"] in current

    # -- migration (Step 3) -------------------------------------------------

    def _migrate(self, generation: int) -> None:
        rng = migration_rng(self.cfg.seed, self.track.name, "b")
        n = self.cfg.evolution.num_islands
        for src in sorted(self.islands):
            island = self.islands[src]
            if not island.population:
                continue
            obj = np.array([r.objectives for r in island.population], dtype=float)
            front = front_1_indices(obj)
            dst = (src + (1 if rng.random() < 0.5 else -1)) % n
            for idx in front:
                rec = island.population[idx]
                new_id = f"{rec.strategy_id}:mig{src}->{dst}:g{generation}"
                import copy as _copy

                migrated = _copy.deepcopy(rec)
                migrated.strategy_id = new_id
                migrated.island = dst
                migrated.generation = generation
                self._register(migrated, self.codes.get(rec.strategy_id, ""))
                self.islands[dst].incoming.append(migrated)
                _copy_migrated_returns(self.run_dir, rec.strategy_id, new_id)
                self.migration_events.append({
                    "generation": generation, "source_island": src,
                    "destination_island": dst, "strategy_id": new_id,
                    "source_strategy_id": rec.strategy_id,
                })
        for island_id, island in self.islands.items():
            if island.incoming:
                self._refresh_island(island_id)
        log.info("[%s Path B] migration event at generation %d", self.track.name, generation)

    def _persist_generation(self, generation: int) -> None:
        self._flag_all_accepted()
        pd.DataFrame(self.pop_rows).to_csv(self.run_dir / "population_history.csv", index=False)
        rows = []
        for island_id, island in self.islands.items():
            for rec in island.population:
                rows.append({**rec.to_pop_row(), "island": island_id, "accepted": True})
        pd.DataFrame(rows).to_csv(self.run_dir / "final_population.csv", index=False)
        if self.migration_events:
            pd.DataFrame(self.migration_events).to_csv(
                self.run_dir / "migration_events.csv", index=False)

    def _flag_all_accepted(self) -> None:
        current = {r.strategy_id for isl in self.islands.values() for r in isl.population}
        for row in self.pop_rows:
            row["accepted"] = row["accepted"] or row["strategy_id"] in current

    # -- final selection discipline (Step 8B) --------------------------------

    def finish(self) -> dict:
        pareto_sets: dict[int, list[dict]] = {}
        combined_pareto: list[str] = []
        test_metrics: dict[str, dict] = {}
        for island_id, island in self.islands.items():
            if not island.population:
                continue
            obj = np.array([r.objectives for r in island.population], dtype=float)
            front = front_1_indices(obj)
            members = [island.population[i] for i in front]
            val_rows = []
            for rec in members:
                m = self._rerun(rec.code, str(self.track.validation.start),
                                str(self.track.validation.end), self.bench_val)
                val_rows.append((rec, objectives_from_metrics(
                    m.sharpe, m.sortino, m.total_return, m.mdd)))
            if not val_rows:
                continue
            val_obj = np.array([row[1] for row in val_rows], dtype=float)
            val_front = front_1_indices(val_obj)
            confirmed = [val_rows[i][0] for i in val_front]
            pareto_sets[island_id] = [
                {"strategy_id": rec.strategy_id, "objectives": OBJECTIVE_NAMES,
                 "values": objectives_from_metrics(
                     rec.metric("sharpe"), rec.metric("sortino"),
                     rec.metric("total_return"), rec.metric("mdd")).tolist()}
                for rec in confirmed
            ]
            combined_pareto.extend(rec.strategy_id for rec in confirmed)
            for rec in confirmed:
                m = self._rerun(rec.code, str(self.track.test.start),
                                str(self.track.test.end), self.bench_test)
                test_metrics[rec.strategy_id] = {
                    "sharpe": m.sharpe, "sortino": m.sortino, "ir": m.ir,
                    "mdd": m.mdd, "total_return": m.total_return,
                    "turnover": m.turnover, "trading_frequency": m.trading_frequency,
                }
        tags = [self.records[sid].style_bits() for sid in combined_pareto if sid in self.records]
        breadth = breadth_from_population(tags) if tags else {"breadth": float("nan"), "n_strategies": 0, "distribution": {}}
        summary = {
            "track": self.track.name,
            "path": "b",
            "tier": self.tier.name,
            "breadth": breadth,
            "validation_confirmed_pareto": {str(k): v for k, v in pareto_sets.items()},
            "n_pareto_members": len(combined_pareto),
            "test": {sid: test_metrics.get(sid, {}) for sid in combined_pareto},
            "n_migration_events": len(self.migration_events),
        }
        with open(self.run_dir / "summary.json", "w") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
        return summary


def _pop_row(rec, accepted: bool) -> dict:
    row = rec.to_pop_row()
    row["accepted"] = accepted
    return row
