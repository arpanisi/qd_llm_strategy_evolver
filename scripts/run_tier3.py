"""Tier 3 full run entrypoint: G=150, 8 candidates/island/gen, M=10, K=50."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evolution.runner import run_tier  # noqa: E402


def main() -> None:
    run_tier("tier3")


if __name__ == "__main__":
    main()
