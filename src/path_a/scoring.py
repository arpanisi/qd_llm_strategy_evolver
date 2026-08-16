"""Steps 8A + 9A — Path A Combined Score, selection discipline, and Breadth.

Combined Score is the Information Ratio alone (the one core metric that is not
a feature-map dimension). Near-ties within ``ir_near_tie`` (0.02) are broken by
lower average Turnover. The validation/test discipline is structural: every
window re-run uses the already-frozen strategy code, and the test-window path
is a separate code path invoked only at the very end.

Breadth (Step 9A) = 1 / sum(p_i^2) over the per-style distribution of occupied
feature-map cells, computed straight from the persisted feature map.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.path_a.archive import ArchiveEntry, FeatureMap


def combined_score(entry: ArchiveEntry) -> float:
    """Step 8A: Combined Score == Information Ratio (alone)."""
    return entry.information_ratio


def better(entry_a: ArchiveEntry, entry_b: ArchiveEntry,
           ir_near_tie: float, turnover_tiebreak: str = "lower") -> ArchiveEntry:
    """Compare two entries by Combined Score with Step 8A's turnover tie-break
    on near-ties (difference within ``ir_near_tie``)."""
    diff = entry_a.combined_score - entry_b.combined_score
    if diff > ir_near_tie:
        return entry_a
    if diff < -ir_near_tie:
        return entry_b
    if turnover_tiebreak == "lower":
        return entry_a if entry_a.turnover < entry_b.turnover else entry_b
    return entry_a if entry_a.turnover > entry_b.turnover else entry_b


def rank_by_score(entries: list[ArchiveEntry],
                  ir_near_tie: float, turnover_tiebreak: str = "lower") -> list[ArchiveEntry]:
    """Rank entries best-first by Combined Score, applying the turnover
    tie-break to near-tied adjacent pairs (Step 4A best-cousins / Step 3
    migration ranking)."""
    ranked = sorted(entries, key=lambda e: -e.combined_score)
    for i in range(len(ranked) - 1):
        a, b = ranked[i], ranked[i + 1]
        if abs(a.combined_score - b.combined_score) <= ir_near_tie and b.turnover < a.turnover:
            ranked[i], ranked[i + 1] = b, a
    return ranked


def best_per_island(fm: FeatureMap, island: int) -> Optional[ArchiveEntry]:
    """Step 8A 'single best strategy per island': among feature-map-occupying
    Archive entries tagged with ``island``, the highest (validation) Combined
    Score. Returns None if the island has no occupant-tagged entries."""
    occupants = [e for e in fm.log if e.placed and e.island == island]
    if not occupants:
        return None
    return max(occupants, key=lambda e: (not np.isnan(e.combined_score), e.combined_score))


def breadth_from_cells(fm: FeatureMap) -> dict:
    """Step 9A: Breadth = 1 / sum(p_i^2) over occupied cells grouped by style
    bit vector, plus the raw per-style distribution."""
    counts: dict[str, int] = {}
    for key, _ in fm.cells.items():
        style = key[0]
        counts[style] = counts.get(style, 0) + 1
    total = sum(counts.values()) or 0
    distribution = {
        style: count / total for style, count in sorted(counts.items())
    }
    breadth = 1.0 / sum(p * p for p in distribution.values()) if total else float("nan")
    return {"breadth": breadth, "n_occupied_cells": total, "distribution": distribution}
