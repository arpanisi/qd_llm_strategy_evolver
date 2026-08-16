"""Steps 8B + 9B — Path B Non-Dominated Sorting, Environmental Selection, Breadth.

Every generation the combined pool (current population + this generation's
offspring + migrated individuals) is sorted front by front on the four
normalized objectives. Fronts fill the next population until the target P=56
would be exceeded; the boundary front is resolved via reference-point niche
selection with deterministic tie-breaking: tied refs are chosen uniformly at
random (seeded), and near-tied distances (within 0.02) are broken by lower
average Turnover.

Breadth (Step 9B) uses the identical formula as Step 9A, applied to the final
validation-confirmed Pareto set across all islands.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from src.path_b.population import perpendicular_distance


def non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """Standard O(M*N^2) NSGA-II fast non-dominated sort; returns a list of
    fronts, each a list of row indices into `objectives`."""
    arr = np.atleast_2d(np.asarray(objectives, dtype=float))
    n = arr.shape[0]
    dominates = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dominates[i, j] = bool(np.all(arr[i] <= arr[j]) and np.any(arr[i] < arr[j]))
    dominated_count = dominates.sum(axis=0)
    front_of = np.full(n, -1, dtype=int)
    fronts: list[list[int]] = []
    remaining = [i for i in range(n)]
    while remaining:
        front = [i for i in remaining if dominated_count[i] == 0]
        if not front:
            raise RuntimeError("non-dominated sort failed to make progress")
        fronts.append(front)
        for i in front:
            front_of[i] = len(fronts) - 1
            for j in remaining:
                if dominates[i, j]:
                    dominated_count[j] -= 1
        remaining = [i for i in remaining if i not in front]
    return fronts


def _associate(normalized_objectives: np.ndarray, refs: np.ndarray) -> np.ndarray:
    dists = perpendicular_distance(normalized_objectives, refs)
    return dists.argmin(axis=1)


def _niche_fill(
    boundary: list[int],
    normalized_objectives: np.ndarray,
    turnover: np.ndarray,
    refs: np.ndarray,
    n_selected_per_ref: dict[int, int],
    target: int,
    rng,
) -> list[int]:
    """Reference-point niche selection for the partial boundary front: prefer
    members associated with refs having the fewest already-accepted
    individuals; tied refs chosen uniformly at random; near-tied distances
    broken by lower Turnover."""
    if not boundary:
        return []
    boundary_obj = normalized_objectives[boundary]
    dists = perpendicular_distance(boundary_obj, refs)
    assoc = dists.argmin(axis=1)
    candidates = list(zip(boundary, [int(r) for r in assoc]))
    accepted: list[int] = []
    while len(accepted) < target and candidates:
        by_ref: dict[int, list[tuple[int, float, float]]] = {}
        for idx, ref in candidates:
            pos = boundary.index(idx)
            by_ref.setdefault(ref, []).append(
                (idx, float(dists[pos, ref]), float(turnover[idx]))
            )
        populated = sorted(by_ref.keys())
        min_count = min(n_selected_per_ref.get(r, 0) for r in populated)
        tied_refs = [r for r in populated if n_selected_per_ref.get(r, 0) == min_count]
        ref = int(rng.choice(tied_refs))
        contenders = sorted(by_ref[ref], key=lambda t: t[1])
        best_dist = contenders[0][1]
        near = [c for c in contenders if c[1] - best_dist <= 0.02]
        winner = min(near, key=lambda t: t[2])[0]
        accepted.append(winner)
        n_selected_per_ref[ref] = n_selected_per_ref.get(ref, 0) + 1
        candidates = [c for c in candidates if c[0] != winner]
    return accepted


def environmental_selection(
    normalized_objectives: np.ndarray,
    turnover: np.ndarray,
    refs: np.ndarray,
    pop_target: int,
    rng,
) -> list[int]:
    """Step 8B: select exactly min(N, pop_target) individuals from the pool.
    Returns row indices into `normalized_objectives` (and `turnover`)."""
    n = normalized_objectives.shape[0]
    if n <= pop_target:
        return list(range(n))
    fronts = non_dominated_sort(normalized_objectives)
    selected: list[int] = []
    for front in fronts:
        if len(selected) + len(front) <= pop_target:
            selected.extend(front)
            continue
        need = pop_target - len(selected)
        sel_arr = normalized_objectives[selected]
        n_selected_per_ref: dict[int, int] = {}
        if len(sel_arr):
            for r in _associate(sel_arr, refs).tolist():
                n_selected_per_ref[int(r)] = n_selected_per_ref.get(int(r), 0) + 1
        selected.extend(
            _niche_fill(front, normalized_objectives, turnover, refs,
                        n_selected_per_ref, need, rng)
        )
        break
    return selected


def front_1_indices(objectives: np.ndarray) -> list[int]:
    return non_dominated_sort(objectives)[0]


def breadth_from_population(style_tags: Iterable[str]) -> dict:
    """Step 9B: Breadth = 1 / sum(p_i^2) over the style-bit distribution of a
    final (validation-confirmed Pareto) population."""
    counts: dict[str, int] = {}
    for tag in style_tags:
        counts[tag] = counts.get(tag, 0) + 1
    total = sum(counts.values()) or 0
    distribution = {s: c / total for s, c in sorted(counts.items())}
    breadth = 1.0 / sum(p * p for p in distribution.values()) if total else float("nan")
    return {"breadth": breadth, "n_strategies": total, "distribution": distribution}
