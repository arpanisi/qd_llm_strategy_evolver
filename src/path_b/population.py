"""Step 2B — Pareto Population Structure and Objectives for Path B (src/path_b).

Four objectives, in minimization form (the standard convention used by the
non-dominated sorting in Step 8B):

    f1 = -Sharpe Ratio
    f2 = -Sortino Ratio
    f3 = -Total Return
    f4 = -Maximum Drawdown      (MDD is <= 0, so -MDD >= 0 is its magnitude)

A fixed set of reference directions is generated once via the Das and Dennis
systematic simplex-lattice method with ``p = 5`` divisions over the 4
objectives, giving C(5+4-1, 4-1) = C(8, 3) = 56 reference points — the
population target size P for every island.

Objective normalization is required before every non-dominated sort /
reference-point association (Steps 4B/8B): a running ideal point
``z* = (min f1, min f2, min f3, min f4)`` is maintained across the whole run
so far, seeded from the 9 seed strategies' objective values (Step 1). The
nadir point is estimated per sort as the worst objective value per dimension
among the current first non-dominated front. Normalization is
``(f - z*) / (nadir - z*)``, with the degenerate-case floor (default 1e-6)
substituted for any denominator below it on that generation only.

Instances are created per track and never shared between tracks.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

N_OBJECTIVES = 4
OBJECTIVE_NAMES = ("f1_neg_sharpe", "f2_neg_sortino", "f3_neg_total_return", "f4_neg_mdd")


def objectives_from_metrics(
    sharpe: float,
    sortino: float,
    total_return: float,
    max_drawdown: float,
) -> np.ndarray:
    """The four Step 2B objectives (minimization form) for one strategy."""
    return np.array([-sharpe, -sortino, -total_return, -max_drawdown], dtype=float)


def das_dennis(num_partitions: int, num_objectives: int) -> np.ndarray:
    """Generate the Das and Dennis systematic simplex-lattice reference points
    over ``num_objectives`` objectives with ``num_partitions`` divisions.

    Returns an (N, M) array of weights, each row summing to 1, with
    N = C(num_partitions + M - 1, M - 1). Each row is the unit simplex point
    that defines one reference direction.
    """
    if num_objectives == 1:
        return np.array([[1.0]], dtype=float)

    def _gen(prefix: tuple[int, ...], remaining: int, slots: int):
        if slots == 1:
            yield prefix + (remaining,)
            return
        for i in range(remaining + 1):
            yield from _gen(prefix + (i,), remaining - i, slots - 1)

    combos = list(_gen((), num_partitions, num_objectives))
    weights = np.asarray(combos, dtype=float) / num_partitions
    return weights


def das_dennis_count(num_partitions: int, num_objectives: int) -> int:
    return math.comb(num_partitions + num_objectives - 1, num_objectives - 1)


class IdealPoint:
    """Running ideal point z* observed across the whole run so far, seeded
    from the 9 seed strategies' objective values (Step 2B)."""

    def __init__(self, n_objectives: int = N_OBJECTIVES) -> None:
        self.n = n_objectives
        self.z_star = np.full(n_objectives, np.inf)
        self.seen = 0

    def observe(self, objectives: Iterable[np.ndarray]) -> None:
        arr = np.atleast_2d(np.asarray(list(objectives), dtype=float))
        if arr.size == 0:
            return
        if arr.shape[1] != self.n:
            raise ValueError(f"expected {self.n} objectives, got {arr.shape[1]}")
        self.z_star = np.minimum(self.z_star, arr.min(axis=0))
        self.seen += arr.shape[0]

    @property
    def is_seeded(self) -> bool:
        return self.seen > 0


def nadir_from_front(front_objectives: np.ndarray) -> np.ndarray:
    """Estimate the nadir point from the current first non-dominated front:
    the worst objective value per dimension (Step 2B)."""
    arr = np.atleast_2d(np.asarray(front_objectives, dtype=float))
    if arr.size == 0:
        raise ValueError("cannot estimate nadir from an empty front")
    return arr.max(axis=0)


def normalize(
    objectives: np.ndarray,
    z_star: np.ndarray,
    nadir: np.ndarray,
    floor: float = 1e-6,
) -> np.ndarray:
    """Normalize objectives to [0, ~1] against the running ideal/nadir:
    ``(f - z*) / (nadir - z*)``. Any denominator below ``floor`` (degenerate
    early-generation front) is replaced with ``floor`` for that objective on
    this generation only — it is never cached into z*/nadir themselves."""
    denom = np.asarray(nadir, dtype=float) - np.asarray(z_star, dtype=float)
    denom = np.where(denom < float(floor), float(floor), denom)
    return (np.asarray(objectives, dtype=float) - np.asarray(z_star, dtype=float)) / denom


def perpendicular_distance(
    objectives: np.ndarray,
    reference_points: np.ndarray,
) -> np.ndarray:
    """Perpendicular distance from each objective point to every reference
    direction: ``|| f - (f . w_hat) w_hat ||_2`` where ``w_hat = w / |w|``
    (Step 4B/8B niche distance). The reference points are simplex weights; the
    direction ray is the unit vector along each, so distances are true
    perpendicular distances to the ray."""
    f = np.atleast_2d(np.asarray(objectives, dtype=float))
    refs = np.asarray(reference_points, dtype=float)
    norms = np.linalg.norm(refs, axis=1, keepdims=True)
    refs_unit = refs / np.where(norms > 0, norms, 1.0)
    dots = f @ refs_unit.T
    proj = dots[:, :, None] * refs_unit[None, :, :]
    return np.linalg.norm(f[:, None, :] - proj, axis=2)  # (n_points, n_refs)


class ReferencePointTable:
    """Live per-island mapping of each reference point to its current occupant
    (nearest by normalized perpendicular distance), recomputed whenever Step 8B
    environmental selection runs. A reference point may have no occupant."""

    def __init__(self, reference_points: np.ndarray) -> None:
        self.reference_points = np.asarray(reference_points, dtype=float)
        self._occupant: dict[int, str] = {}
        self._distance: dict[int, float] = {}

    def recompute(
        self,
        individuals: list[str],
        normalized_objectives: np.ndarray,
    ) -> None:
        self._occupant = {}
        self._distance = {}
        norm_obj = np.atleast_2d(np.asarray(normalized_objectives, dtype=float))
        dists = perpendicular_distance(norm_obj, self.reference_points)
        for i, ind_id in enumerate(individuals):
            ref_idx = int(dists[i].argmin())
            d = float(dists[i][ref_idx])
            if ref_idx not in self._occupant or d < self._distance[ref_idx]:
                self._occupant[ref_idx] = ind_id
                self._distance[ref_idx] = d

    def occupant_of(self, ref_index: int) -> Optional[str]:
        return self._occupant.get(ref_index)

    def occupied_count(self) -> int:
        return len(self._occupant)

    def as_dict(self) -> dict[int, str]:
        return dict(self._occupant)


class PopulationHistoryLog:
    """The Path B equivalent of Path A's Archive Log: records every strategy
    ever generated (accepted into the population or not) with full metadata —
    the four objective values plus Sharpe/Sortino/Return/MDD/Frequency/
    Turnover/Style/lineage (Step 2B)."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def add(
        self,
        *,
        strategy_id: str,
        island: int,
        generation: int,
        sharpe: float,
        sortino: float,
        total_return: float,
        max_drawdown: float,
        trading_frequency: float,
        turnover: float,
        style: str,
        lineage_parents: list[str],
        lineage_cousins: list[str],
        accepted: bool = False,
        **extra: Any,
    ) -> None:
        objectives = objectives_from_metrics(sharpe, sortino, total_return, max_drawdown)
        record = {
            "strategy_id": strategy_id,
            "island": island,
            "generation": generation,
            "f1_neg_sharpe": objectives[0],
            "f2_neg_sortino": objectives[1],
            "f3_neg_total_return": objectives[2],
            "f4_neg_mdd": objectives[3],
            "sharpe": sharpe,
            "sortino": sortino,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "trading_frequency": trading_frequency,
            "turnover": turnover,
            "style": style,
            "lineage_parents": "|".join(lineage_parents),
            "lineage_cousins": "|".join(lineage_cousins),
            "accepted": accepted,
        }
        record.update(extra)
        self._records.append(record)

    def __len__(self) -> int:
        return len(self._records)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._records)
