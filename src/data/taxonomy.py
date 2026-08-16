"""Strategy-style taxonomy (Step 1) and island mapping (Step 3).

The 8 style categories are fixed, in order, and encoded as an 8-bit binary
vector: leftmost bit = category 1 (Trend-Following), rightmost = category 8
(Calendar/Seasonal). Island `i` is initialized with seed strategy `i`; the
benchmark island is number 8.
"""

from __future__ import annotations

from dataclasses import dataclass

STYLE_NAMES: tuple[str, ...] = (
    "trend_following",
    "mean_reversion",
    "volatility",
    "volume_liquidity",
    "breakout",
    "statistical_arb",
    "risk_parity",
    "calendar_seasonal",
)

N_STYLES = len(STYLE_NAMES)

# Benchmark island (index 8 in the 9-island ring) is not a taxonomy category.
NUM_ISLANDS = N_STYLES + 1
BENCHMARK_ISLAND = N_STYLES


@dataclass(frozen=True)
class StyleVector:
    """An 8-bit style membership vector."""

    bits: tuple[bool, ...]

    def __init__(self, *bits: bool):
        if len(bits) != N_STYLES:
            raise ValueError(f"style vector must have exactly {N_STYLES} bits")
        object.__setattr__(self, "bits", tuple(bool(b) for b in bits))

    @classmethod
    def single(cls, index: int) -> "StyleVector":
        if not 0 <= index < N_STYLES:
            raise ValueError(f"style index out of range: {index}")
        return cls(*(i == index for i in range(N_STYLES)))

    @classmethod
    def from_binary_string(cls, value: str) -> "StyleVector":
        value = value.strip()
        if len(value) != N_STYLES or any(ch not in "01" for ch in value):
            raise ValueError(f"expected an {N_STYLES}-char 0/1 string, got {value!r}")
        return cls(*(ch == "1" for ch in value))

    @classmethod
    def from_names(cls, names: list[str]) -> "StyleVector":
        name_set = set(names)
        return cls(*(name in name_set for name in STYLE_NAMES))

    def as_binary_string(self) -> str:
        return "".join("1" if b else "0" for b in self.bits)

    def names(self) -> list[str]:
        return [STYLE_NAMES[i] for i, b in enumerate(self.bits) if b]

    def flip_bits(self, rng, k: int) -> "StyleVector":
        """Flip ``k`` random bits (repeats can cancel; intentional noise)."""
        new_bits = list(self.bits)
        for _ in range(k):
            i = int(rng.integers(0, N_STYLES))
            new_bits[i] = not new_bits[i]
        return StyleVector(*new_bits)

    def __hash__(self) -> int:
        return hash(self.bits)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, StyleVector) and self.bits == other.bits

    def __repr__(self) -> str:
        return f"StyleVector({self.as_binary_string()})"


# Each island's identity style (0..7 = the taxonomy category, 8 = benchmark).
ISLAND_STYLES: tuple[int, ...] = tuple(range(NUM_ISLANDS))
