"""Step 2A/2B setup entrypoint: freeze Path A's feature-map bin edges and seed
Path B's running ideal point from the 9 seed strategies' train-window metrics
(Step 1), independently per track.

For each track it:
  * freezes the 5 continuous feature-map dimensions' bin edges (observed
    seed min/max expanded by path_a.bin_headroom each side, cut into
    path_a.bins_per_dim equal-width bins — never recomputed afterwards);
  * places the 9 seed entries into the feature map (Archive Log + cells);
  * generates the fixed Das-Dennis reference directions (p=5, M=4 -> 56);
  * seeds the running ideal point z* from the seeds' objective values;
  * records the seeds in the Population History Log.

Persists everything under outputs/search/<track>/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config.settings import TrackConfig, load_config
from src.path_a.archive import (
    DIM_COLUMNS,
    FeatureMap,
    build_seed_entries,
    freeze_bin_edges,
)
from src.path_b.population import (
    N_OBJECTIVES,
    OBJECTIVE_NAMES,
    IdealPoint,
    das_dennis,
    das_dennis_count,
    objectives_from_metrics,
    PopulationHistoryLog,
)

OUTPUTS = ROOT / "outputs" / "search"


def _seeds(track: TrackConfig) -> pd.DataFrame:
    path = ROOT / "outputs" / "seeds" / track.name / "seeds.csv"
    return pd.read_csv(path)


def setup_track(track: TrackConfig, bins_per_dim: int, headroom: float,
                ir_near_tie: float, turnover_tiebreak: str,
                das_dennis_p: int, pop_target: int) -> dict:
    seeds = _seeds(track)

    # --- Step 2A: freeze bin edges from the 9 seeds' train metrics ---------
    frozen = freeze_bin_edges(seeds, bins_per_dim=bins_per_dim, headroom=headroom)

    entries = build_seed_entries(seeds)
    fm = FeatureMap(frozen, ir_near_tie=ir_near_tie, turnover_tiebreak=turnover_tiebreak)
    placements = {}
    for e in entries:
        placed, reason = fm.place(e)
        placements[e.strategy_name] = {"placed": placed, "reason": reason}

    # --- Step 2B: Das-Dennis reference directions + ideal point ------------
    refs = das_dennis(das_dennis_p, N_OBJECTIVES)
    n_expected = das_dennis_count(das_dennis_p, N_OBJECTIVES)
    assert refs.shape == (n_expected, N_OBJECTIVES), (
        f"expected {n_expected}x{N_OBJECTIVES} reference points, got {refs.shape}"
    )
    assert np.allclose(refs.sum(axis=1), 1.0), "reference points must lie on the simplex"

    ideal = IdealPoint(N_OBJECTIVES)
    for e in entries:
        ideal.observe([objectives_from_metrics(
            e.sharpe, e.sortino, e.total_return, e.max_drawdown
        )])

    pop_log = PopulationHistoryLog()
    for e in entries:
        pop_log.add(
            strategy_id=e.entry_id,
            island=e.island,
            generation=e.generation,
            sharpe=e.sharpe,
            sortino=e.sortino,
            total_return=e.total_return,
            max_drawdown=e.max_drawdown,
            trading_frequency=e.trading_frequency,
            turnover=e.turnover,
            style=e.style.as_binary_string(),
            lineage_parents=e.lineage_parents,
            lineage_cousins=e.lineage_cousins,
            accepted=True,
        )

    # --- persist ------------------------------------------------------------
    out = OUTPUTS / track.name
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "bin_edges.json", "w") as fh:
        json.dump(
            {dim: frozen[dim].to_dict() for dim in frozen},
            fh, indent=2, sort_keys=True,
        )
    with open(out / "ideal_point.json", "w") as fh:
        json.dump({
            "z_star": ideal.z_star.tolist(),
            "objective_names": list(OBJECTIVE_NAMES),
            "n_seeds": ideal.seen,
            "pop_target": pop_target,
        }, fh, indent=2, sort_keys=True)

    refs_df = pd.DataFrame(refs, columns=list(OBJECTIVE_NAMES))
    refs_df.insert(0, "ref_index", range(len(refs_df)))
    refs_df.to_csv(out / "reference_points.csv", index=False)

    fm.log.to_dataframe().to_csv(out / "archive_log.csv", index=False)
    fm.cells_dataframe().to_csv(out / "archive_cells.csv", index=False)
    pop_log.to_dataframe().to_csv(out / "population_history.csv", index=False)

    summary = {
        "track": track.name,
        "frozen_bin_edges": {dim: frozen[dim].to_dict() for dim in frozen},
        "placements": placements,
        "occupied_cells": fm.cell_count(),
        "archive_log_entries": len(fm.log),
        "reference_points": int(refs.shape[0]),
        "reference_points_expected": n_expected,
        "ideal_point_z_star": ideal.z_star.tolist(),
        "population_history_entries": len(pop_log),
    }
    with open(out / "setup_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    return summary


def main() -> None:
    cfg = load_config()
    print(f"search setup outputs written to {OUTPUTS}\n")
    for track in cfg.tracks:
        s = setup_track(
            track,
            bins_per_dim=cfg.path_a.bins_per_dim,
            headroom=cfg.path_a.bin_headroom,
            ir_near_tie=cfg.path_a.ir_near_tie,
            turnover_tiebreak=cfg.path_a.turnover_tiebreak,
            das_dennis_p=cfg.path_b.das_dennis_p,
            pop_target=cfg.path_b.pop_target,
        )
        print(f"=== {track.name} ===")
        print("  frozen bin edges (observed -> expanded, per dim):")
        for dim, col in DIM_COLUMNS.items():
            b = s["frozen_bin_edges"][dim]
            print(
                f"    {col:16s} seeds [{b['observed_min']: .4f}, "
                f"{b['observed_max']: .4f}] -> frozen [{b['low']: .4f}, "
                f"{b['high']: .4f}] in {b['bins']} bins"
            )
        print(f"  feature map: {s['occupied_cells']} occupied cells, "
              f"{s['archive_log_entries']} archive entries")
        for name, p in s["placements"].items():
            print(f"    {name:32s} {p['reason']}")
        print(f"  reference points: {s['reference_points']} "
              f"(expected {s['reference_points_expected']})")
        print(f"  ideal point z*: "
              + ", ".join(f"{n}={v:.4f}" for n, v in zip(OBJECTIVE_NAMES, s["ideal_point_z_star"])))
        print()


if __name__ == "__main__":
    main()
