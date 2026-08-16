"""Seeded RNG streams for a fully reproducible run.

A master ``default_rng(run.seed)`` derives one independent stream per
(track, path, island) plus one for cross-island migration events. Every draw
in Steps 3/4A/4B/8B — island choice, parent/cousin sampling, migration target,
reference-point tie-breaking — goes through these streams so a given run seed
reproduces bit-for-bit.
"""

from __future__ import annotations

import numpy as np


def _splitmix(seed: int, index: int) -> int:
    x = (seed + 0x9E3779B97F4A7C15 * (index + 1)) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return (x ^ (x >> 31)) & 0xFFFFFFFFFFFFFFFF


def island_rng(run_seed: int, track: str, path: str, island: int) -> np.random.Generator:
    return np.random.default_rng(_splitmix(run_seed, hash((track, path, island)) & 0xFFFFFFFF))


def migration_rng(run_seed: int, track: str, path: str) -> np.random.Generator:
    return np.random.default_rng(_splitmix(run_seed, hash((track, path, "migration")) & 0xFFFFFFFF))
