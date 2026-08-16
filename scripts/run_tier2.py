"""Tier 2 small-test entrypoint: G=8, 2 candidates/island/gen, M=4, K=8."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evolution.runner import run_tier  # noqa: E402


def main() -> None:
    run_tier("tier2")


if __name__ == "__main__":
    main()
