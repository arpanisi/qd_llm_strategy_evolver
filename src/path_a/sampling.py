"""Step 4A — Parent and Cousin Sampling [Path A].

One parent + up to seven cousins are drawn for a given island before Step 5.
  * Parent: two-stage weighted draw — with prob alpha the "best parent" branch
    (uniform over this island's feature-map occupants), else the "diverse"
    branch (uniform over all this island's Archive entries). This yields
    P(parent=s) = alpha/|M_I| + (1-alpha)/|I| for occupants, else (1-alpha)/|I|.
  * Cousins: 2 best (top Combined Score with turnover tie-break), 3 diverse
    (simultaneous 6-dimensional neighbor perturbation of the parent's cell,
    10-attempt resample cap, random-cousin substitution fallback), 2 random.
  * Early generations: sample with replacement when the island has fewer than
    8 strategies so 1 parent + 7 cousins can always be produced.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.config.settings import PathAConfig
from src.path_a.archive import ContinuousBins, FeatureMap
from src.path_a.scoring import rank_by_score

_N_NEIGHBOR_ATTEMPTS = 10


def _bin_neighbor(bin_idx: int, rng, bins: ContinuousBins, sigma: float) -> int:
    """floor(Normal(parent_bin, sigma^2)) clamped into [0, bins-1]."""
    raw = int(np.floor(rng.normal(loc=bin_idx, scale=sigma)))
    return min(max(raw, 0), bins.bins - 1)


def _neighbor_cell(fm: FeatureMap, parent_style_bits: str, parent_bins: dict[str, int],
                   rng, sigma: float, k_bitflips: int) -> tuple[str, ...]:
    """One 6-dimensional neighbor cell key: all continuous dimensions perturbed
    simultaneously plus k random style bit-flips."""
    from src.data.taxonomy import StyleVector

    style = StyleVector.from_binary_string(parent_style_bits).flip_bits(rng, k_bitflips)
    bins = {}
    for dim, cb in fm.frozen.items():
        bins[dim] = _bin_neighbor(parent_bins[dim], rng, cb, sigma)
    return (style.as_binary_string(), *(
        str(bins[dim]) for dim in fm.frozen
    ))


class PathASampler:
    def __init__(self, fm: FeatureMap, cfg: PathAConfig, rng) -> None:
        self.fm = fm
        self.cfg = cfg
        self.rng = rng

    # -- parent ------------------------------------------------------------

    def _island_entries(self, island: int) -> list:
        return [e for e in self.fm.log if e.island == island]

    def _island_occupants(self, island: int) -> list:
        return [e for e in self.fm.log if e.island == island and e.placed]

    def sample_parent(self, island: int):
        """Two-stage parent draw; returns an Archive entry."""
        all_entries = self._island_entries(island)
        occupants = self._island_occupants(island)
        if not all_entries:
            raise RuntimeError(f"island {island} has no Archive entries to sample from")
        if self.rng.random() < self.cfg.alpha and occupants:
            return occupants[self.rng.integers(0, len(occupants))]
        return all_entries[self.rng.integers(0, len(all_entries))]

    # -- cousins -----------------------------------------------------------

    def sample_cousins(self, island: int, parent) -> list:
        """Return up to 7 cousins (2 best + 3 diverse + 2 random)."""
        entries = self._island_entries(island)
        cousins: list = []
        ranked = rank_by_score(entries, self.fm.ir_near_tie, self.fm.turnover_tiebreak)
        if ranked:
            # with-replacement fallback when the island has fewer strategies
            # than the slot count (Step 4A early-generation rule)
            cousins.extend(ranked[i % len(ranked)] for i in range(self.cfg.n_best_cousins))
        cousins.extend(self._diverse_cousins(parent, entries, self.cfg.n_diverse_cousins))
        for _ in range(self.cfg.n_random_cousins):
            cousins.append(entries[self.rng.integers(0, len(entries))])
        return cousins

    def _diverse_cousins(self, parent, entries: list, n: int) -> list:
        parent_bins = {
            dim: self.fm.frozen[dim].bin_index(getattr(parent, dim))
            for dim in self.fm.frozen
        }
        parent_style = parent.style.as_binary_string()
        chosen: list = []
        while len(chosen) < n:
            found = None
            for _ in range(_N_NEIGHBOR_ATTEMPTS):
                key = _neighbor_cell(
                    self.fm, parent_style, parent_bins, self.rng,
                    self.cfg.cousin_neighbor_sigma, self.cfg.bitflips,
                )
                if key in self.fm.cells:
                    found = self.fm.cells[key]
                    break
            if found is None:
                # substitute a random cousin per Step 4A
                found = entries[self.rng.integers(0, len(entries))]
            chosen.append(found)
        return chosen
