"""Tier 1 smoke-test entrypoint: G=1, 1 candidate/island/gen, no migration,
no curation. Runs both paths on both tracks.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evolution.runner import run_tier  # noqa: E402


def main() -> None:
    results = run_tier("tier1")
    print("\nTier 1 complete:")
    for track, paths in results.items():
        for path, summary in paths.items():
            cells = summary.get("occupied_cells", summary.get("breadth", {}).get("n_strategies"))
            print(f"  {track} / path {path}: breadths="
                  f"{summary.get('breadth', {}).get('breadth')}")


if __name__ == "__main__":
    main()
