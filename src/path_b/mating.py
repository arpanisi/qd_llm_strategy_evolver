"""Step 4B — Mating Selection [Path B].

One parent via binary tournament: two individuals drawn uniformly from the
island's current population; the winner is the one with the better
(lower-numbered) front rank from the most recent sort, and on the same front
the one with the smaller normalized perpendicular distance to its associated
reference point. Seven cousins come from the parent's nearest reference
directions (by cosine similarity), outward search past unoccupied directions,
with Path A's with-replacement fallback when the island has fewer than 8
individuals.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.path_b.population import ReferencePointTable, perpendicular_distance


class PathBMating:
    def __init__(self, refs: np.ndarray, rng) -> None:
        self.refs = np.asarray(refs, dtype=float)
        self.rng = rng

    def _front_ranks(self, normalized: np.ndarray) -> np.ndarray:
        from src.path_b.selection import non_dominated_sort

        ranks = np.full(normalized.shape[0], -1, dtype=int)
        for rank, front in enumerate(non_dominated_sort(normalized)):
            for i in front:
                ranks[i] = rank
        return ranks

    def sample_parent_and_cousins(
        self,
        population: list,
        normalized_objectives: np.ndarray,
        ref_table: ReferencePointTable,
        id_to_index: dict,
        n_cousins: int = 7,
    ) -> tuple[Optional[object], list[object]]:
        """`population` is the island's current individuals (any objects);
        `normalized_objectives` are their normalized objectives; `ref_table`
        maps each reference point to its occupant's strategy id; `id_to_index`
        maps id -> index into `population`. Returns (parent, cousins) with the
        early-generation with-replacement fallback."""
        n = len(population)
        if n == 0:
            raise RuntimeError("cannot mate from an empty population")
        if n == 1:
            parent = population[0]
            return parent, [parent] * min(n_cousins, 7)

        ranks = self._front_ranks(normalized_objectives)
        dists = perpendicular_distance(normalized_objectives, self.refs)
        assoc = dists.argmin(axis=1)

        def _tournament() -> object:
            a = int(self.rng.integers(0, n))
            b = int(self.rng.integers(0, n))
            while b == a:
                b = int(self.rng.integers(0, n))
            if ranks[a] != ranks[b]:
                return population[a if ranks[a] < ranks[b] else b]
            return population[a if dists[a, assoc[a]] <= dists[b, assoc[b]] else b]

        parent_idx = population.index(_tournament())
        parent = population[parent_idx]
        parent_ref = int(assoc[parent_idx])

        # cousin directions: nearest refs by cosine similarity, excluding own
        ref_norms = np.linalg.norm(self.refs, axis=1, keepdims=True)
        cos = (self.refs @ self.refs.T) / (ref_norms @ ref_norms.T)
        order = np.argsort(-cos[parent_ref])[1:]  # nearest first, skipping self

        cousins: list[object] = []
        seen: set = set()
        for ref_idx in order:
            if len(cousins) >= n_cousins:
                break
            occ = ref_table.occupant_of(int(ref_idx))
            if occ is None:
                continue  # outward search
            idx = id_to_index.get(occ)
            if idx is None or idx in seen:
                continue
            seen.add(idx)
            cousins.append(population[idx])

        while len(cousins) < n_cousins:
            cousins.append(population[int(self.rng.integers(0, n))])
        return parent, cousins[:n_cousins]
