#!/usr/bin/env python
"""Fetch and materialize both tracks' data (Step 1).

Usage:
    python scripts/fetch_data.py [--track equities|futures|all] [--skip-cache]
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data.equities import EQUITIES_PARQUET, materialize_equities
from src.data.futures import FUTURES_PARQUET, materialize_futures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["equities", "futures", "all"], default="all")
    parser.add_argument("--force", action="store_true", help="re-fetch even if cached")
    args = parser.parse_args()
    cfg = load_config()

    if args.track in ("equities", "all"):
        if EQUITIES_PARQUET.exists() and not args.force:
            print(f"equities cache exists: {EQUITIES_PARQUET} (use --force to refetch)")
        else:
            universe = materialize_equities(cfg)
            print(f"equities: {len(universe)} names -> {EQUITIES_PARQUET}")

    if args.track in ("futures", "all"):
        if FUTURES_PARQUET.exists() and not args.force:
            print(f"futures cache exists: {FUTURES_PARQUET} (use --force to refetch)")
        else:
            result = materialize_futures(cfg)
            print(f"futures: {result} -> {FUTURES_PARQUET}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
