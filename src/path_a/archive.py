"""Step 2A — Quality-Diversity Archive (Feature Map) for Path A (src/path_a).

The archive is a multi-dimensional grid over:
  1. Style category (8-bit binary vector from Step 1 — categorical, not binned)
  2. Trading Frequency  (continuous, equal-width bins)
  3. Maximum Drawdown   (continuous, equal-width bins)
  4. Sharpe Ratio       (continuous, equal-width bins)
  5. Sortino Ratio      (continuous, equal-width bins)
  6. Total Return       (continuous, equal-width bins)

Bin edges are frozen once, immediately after the 9 seed strategies (Step 1)
are backtested over the training window: take the observed min/max of each
continuous metric across those seeds, expand by ``headroom`` (default 50%) in
each direction, and cut that expanded range into ``bins_per_dim`` equal-width
bins (default 16). Edges never change for the rest of the run; values outside
the frozen range are clamped into the nearest edge bin.

Each cell stores a reference to exactly one Archive Log entry — the best one
found so far for that combination, measured by Combined Score (= Information
Ratio, Step 8A), with Step 8A's turnover tie-break when scores are within the
near-tie threshold. Every strategy ever generated is retained in the flat
Archive Log (winners and losers) with full metadata. Instances are created per
track and never shared between tracks.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from src.data.taxonomy import N_STYLES, StyleVector

# Dimension names, in the Step 2A order.
CONTINUOUS_DIMS: tuple[str, ...] = (
    "trading_frequency",
    "max_drawdown",
    "sharpe",
    "sortino",
    "total_return",
)

# maps each dimension to the seeds.csv column label
DIM_COLUMNS: dict[str, str] = {
    "trading_frequency": "Trading Frequency",
    "max_drawdown": "Max Drawdown",
    "sharpe": "Sharpe Ratio",
    "sortino": "Sortino Ratio",
    "total_return": "Total Return",
}


@dataclass(frozen=True)
class ContinuousBins:
    """Frozen equal-width bin edges for one continuous feature-map dimension."""

    name: str
    low: float
    high: float
    bins: int
    edges: tuple[float, ...]
    observed_min: float
    observed_max: float

    @property
    def width(self) -> float:
        return (self.high - self.low) / self.bins

    def bin_index(self, value: float) -> int:
        """Index of the bin containing `value`, clamping into the nearest edge
        bin when `value` falls outside the frozen range (Step 2A)."""
        if not np.isfinite(value):
            return 0
        if value <= self.low:
            return 0
        if value >= self.high:
            return self.bins - 1
        idx = int((value - self.low) / self.width)
        return min(max(idx, 0), self.bins - 1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def freeze_bin_edges(
    seed_metrics: pd.DataFrame,
    bins_per_dim: int,
    headroom: float = 0.5,
) -> dict[str, ContinuousBins]:
    """Compute the frozen per-dimension bin edges from the 9 seeds' train
    metrics (Step 2A). `seed_metrics` columns are the DIM_COLUMNS labels."""
    frozen: dict[str, ContinuousBins] = {}
    for dim in CONTINUOUS_DIMS:
        col = DIM_COLUMNS[dim]
        values = seed_metrics[col].astype(float)
        lo = float(values.min())
        hi = float(values.max())
        span = hi - lo
        if span <= 0 or not np.isfinite(span):
            span = max(1.0, abs(hi) * 0.5)
            lo, hi = hi, hi + span
        low = lo - headroom * span
        high = hi + headroom * span
        edges = tuple(np.linspace(low, high, bins_per_dim + 1).tolist())
        frozen[dim] = ContinuousBins(
            name=dim,
            low=low,
            high=high,
            bins=bins_per_dim,
            edges=edges,
            observed_min=lo,
            observed_max=hi,
        )
    return frozen


@dataclass
class ArchiveEntry:
    """One Archive Log entry: a strategy plus everything needed to place it in
    the feature map and trace it. `combined_score` is the Information Ratio
    (Step 8A); `turnover` is Step 10's average one-way turnover."""

    entry_id: str
    island: int
    generation: int
    strategy_name: str
    style: StyleVector
    trading_frequency: float
    max_drawdown: float
    sharpe: float
    sortino: float
    total_return: float
    turnover: float
    information_ratio: float
    lineage_parents: list[str] = field(default_factory=list)
    lineage_cousins: list[str] = field(default_factory=list)
    source: str = ""
    placed: bool = False
    placed_reason: str = ""

    @property
    def combined_score(self) -> float:
        return self.information_ratio

    def cell_components(self, frozen: dict[str, ContinuousBins]) -> tuple[str, ...]:
        return (
            self.style.as_binary_string(),
            *(
                str(frozen[dim].bin_index(getattr(self, dim)))
                for dim in CONTINUOUS_DIMS
            ),
        )


class ArchiveLog:
    """Flat, append-only record of every strategy ever generated (winners and
    losers) for one track. Supports the Step 9A Breadth queries without any
    backtest re-run."""

    def __init__(self) -> None:
        self._entries: list[ArchiveEntry] = []

    def add(self, entry: ArchiveEntry) -> None:
        self._entries.append(entry)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def entries_in_style(self, style: StyleVector) -> list[ArchiveEntry]:
        return [e for e in self._entries if e.style == style]

    def entries_on_island(self, island: int) -> list[ArchiveEntry]:
        return [e for e in self._entries if e.island == island]

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for e in self._entries:
            row = asdict(e)
            row["style"] = e.style.as_binary_string()
            rows.append(row)
        return pd.DataFrame(rows)


class FeatureMap:
    """The Path A archive: a sparse grid of occupied cells plus the flat
    Archive Log. One instance per track, built from that track's frozen bins."""

    def __init__(
        self,
        frozen: dict[str, ContinuousBins],
        ir_near_tie: float = 0.02,
        turnover_tiebreak: str = "lower",
    ) -> None:
        self.frozen = frozen
        self.ir_near_tie = ir_near_tie
        self.turnover_tiebreak = turnover_tiebreak
        self.cells: dict[tuple[str, ...], ArchiveEntry] = {}
        self.log = ArchiveLog()

    # -- placement ---------------------------------------------------------

    def place(self, entry: ArchiveEntry) -> tuple[bool, str]:
        """Try to put `entry` into its feature-map cell. The entry is always
        retained in the Archive Log; the cell keeps the best Combined Score
        (IR) so far, with the Step 8A turnover tie-break on near-ties."""
        self.log.add(entry)
        key = entry.cell_components(self.frozen)
        incumbent = self.cells.get(key)

        if incumbent is None:
            self.cells[key] = entry
            entry.placed, entry.placed_reason = True, "cell empty"
            return True, "placed: empty cell"

        diff = entry.combined_score - incumbent.combined_score
        if diff > self.ir_near_tie:
            self.cells[key] = entry
            entry.placed, entry.placed_reason = True, "beat incumbent"
            return True, "placed: beat incumbent on Combined Score"
        if diff < -self.ir_near_tie:
            entry.placed, entry.placed_reason = False, "lost to incumbent"
            return False, "rejected: incumbent has better Combined Score"

        # Near tie: Step 8A turnover tie-break decides the winner.
        if self.turnover_tiebreak == "lower":
            winner = entry.turnover < incumbent.turnover
        elif self.turnover_tiebreak == "higher":
            winner = entry.turnover > incumbent.turnover
        else:  # pragma: no cover - config is validated upstream
            raise ValueError(f"unknown turnover tie-break: {self.turnover_tiebreak}")
        if winner:
            self.cells[key] = entry
            entry.placed, entry.placed_reason = True, "won turnover tie-break"
            return True, "placed: won near-tie on turnover"
        entry.placed, entry.placed_reason = False, "lost turnover tie-break"
        return False, "rejected: lost near-tie on turnover"

    # -- queries -----------------------------------------------------------

    def occupied_cells(self) -> list[tuple[str, ArchiveEntry]]:
        return sorted((key, e) for key, e in self.cells.items())

    def cell_count(self) -> int:
        return len(self.cells)

    def strategies_in_style(self, style: StyleVector) -> list[ArchiveEntry]:
        return self.log.entries_in_style(style)

    def strategies_on_island(self, island: int) -> list[ArchiveEntry]:
        return self.log.entries_on_island(island)

    def cells_dataframe(self) -> pd.DataFrame:
        rows = []
        for key, e in self.cells.items():
            rows.append(
                {
                    "cell_key": "|".join(key),
                    "style": key[0],
                    "trading_frequency_bin": int(key[1]),
                    "max_drawdown_bin": int(key[2]),
                    "sharpe_bin": int(key[3]),
                    "sortino_bin": int(key[4]),
                    "total_return_bin": int(key[5]),
                    "entry_id": e.entry_id,
                    "island": e.island,
                    "generation": e.generation,
                    "information_ratio": e.information_ratio,
                    "turnover": e.turnover,
                }
            )
        return pd.DataFrame(rows)


def build_seed_entries(
    seeds_df: pd.DataFrame,
    *,
    generation: int = 0,
) -> list[ArchiveEntry]:
    """Turn a track's seeds.csv rows (Step 1) into Archive Log entries.

    ``seed_<i>_<style>`` maps to island `i`; seeds 0..7 carry the single-bit
    taxonomy style `i`, and the benchmark seed (island 8) carries the all-zero
    style vector (it is not a taxonomy category).
    """
    entries: list[ArchiveEntry] = []
    for _, row in seeds_df.iterrows():
        stem = str(row["seed"])
        island = int(stem.split("_")[1])
        style = (
            StyleVector.single(island)
            if island < N_STYLES
            else StyleVector(*(False,) * N_STYLES)
        )
        entries.append(
            ArchiveEntry(
                entry_id=f"{stem}:g{generation}",
                island=island,
                generation=generation,
                strategy_name=stem,
                style=style,
                trading_frequency=float(row["Trading Frequency"]),
                max_drawdown=float(row["Max Drawdown"]),
                sharpe=float(row["Sharpe Ratio"]),
                sortino=float(row["Sortino Ratio"]),
                total_return=float(row["Total Return"]),
                turnover=float(row["Turnover"]),
                information_ratio=float(row["Information Ratio"]),
                lineage_parents=[],
                lineage_cousins=[],
            )
        )
    return entries
