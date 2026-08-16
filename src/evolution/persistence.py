"""Load/save helpers bridging Step 2A/2B setup artifacts and the run engine.

Rebuilds the per-track FeatureMap (frozen bins + seed placements), the Das-
Dennis reference directions, and the seeded IdealPoint from the persisted
outputs/search/<track>/ files so an evolution run continues exactly where
setup_search.py left off.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.path_a.archive import ContinuousBins, FeatureMap
from src.path_b.population import IdealPoint

OUTPUTS = Path(__file__).resolve().parents[2] / "outputs" / "search"


def load_feature_map(track_name: str, ir_near_tie: float,
                     turnover_tiebreak: str = "lower") -> FeatureMap:
    d = OUTPUTS / track_name
    with open(d / "bin_edges.json") as fh:
        raw = json.load(fh)
    frozen = {
        dim: ContinuousBins(
            name=data["name"], low=float(data["low"]), high=float(data["high"]),
            bins=int(data["bins"]), edges=tuple(float(e) for e in data["edges"]),
            observed_min=float(data["observed_min"]),
            observed_max=float(data["observed_max"]),
        )
        for dim, data in raw.items()
    }
    fm = FeatureMap(frozen, ir_near_tie=ir_near_tie, turnover_tiebreak=turnover_tiebreak)
    cells = pd.read_csv(d / "archive_cells.csv", dtype=str)
    log = pd.read_csv(d / "archive_log.csv", dtype=str)
    fm.log._entries = [_row_to_entry(row) for _, row in log.iterrows()]
    for _, row in cells.iterrows():
        key = tuple(
            str(row[c]) for c in [
                "style", "trading_frequency_bin", "max_drawdown_bin",
                "sharpe_bin", "sortino_bin", "total_return_bin",
            ]
        )
        entry_id = str(row["entry_id"])
        fm.cells[key] = next(e for e in fm.log if e.entry_id == entry_id)
    return fm


def _row_to_entry(row: pd.Series):
    from src.data.taxonomy import StyleVector
    from src.path_a.archive import ArchiveEntry

    def _f(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _split(value):
        if value is None or str(value).strip() in ("", "nan"):
            return []
        return [x for x in str(value).split("|") if x]

    return ArchiveEntry(
        entry_id=str(row["entry_id"]),
        island=int(row["island"]),
        generation=int(row["generation"]),
        strategy_name=str(row["strategy_name"]),
        style=StyleVector.from_binary_string(str(row["style"])),
        trading_frequency=_f(row["trading_frequency"]),
        max_drawdown=_f(row["max_drawdown"]),
        sharpe=_f(row["sharpe"]),
        sortino=_f(row["sortino"]),
        total_return=_f(row["total_return"]),
        turnover=_f(row["turnover"]),
        information_ratio=_f(row["information_ratio"]),
        lineage_parents=_split(row.get("lineage_parents")),
        lineage_cousins=_split(row.get("lineage_cousins")),
        source=str(row.get("source", "")) if row.get("source") != "nan" else "",
        placed=str(row.get("placed", "")).strip().lower() == "true",
        placed_reason=str(row.get("placed_reason", "")),
    )


def load_references(track_name: str):
    df = pd.read_csv(OUTPUTS / track_name / "reference_points.csv")
    return df[["f1_neg_sharpe", "f2_neg_sortino", "f3_neg_total_return", "f4_neg_mdd"]].to_numpy()


def load_ideal_point(track_name: str) -> IdealPoint:
    with open(OUTPUTS / track_name / "ideal_point.json") as fh:
        raw = json.load(fh)
    ideal = IdealPoint(n_objectives=len(raw["z_star"]))
    ideal.z_star = __import__("numpy").asarray(raw["z_star"], dtype=float)
    ideal.seen = int(raw["n_seeds"])
    return ideal
