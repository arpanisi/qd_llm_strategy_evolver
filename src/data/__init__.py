"""Lazy re-export facade for the data layer.

Fetch-time dependencies (e.g. wrds, yfinance) are only imported on demand so
that importing subpackages like ``src.data.seeds`` in backtest tooling does
not require them.
"""

from __future__ import annotations

from src.data.taxonomy import (
    BENCHMARK_ISLAND,
    ISLAND_STYLES,
    N_STYLES,
    NUM_ISLANDS,
    STYLE_NAMES,
    StyleVector,
)

__all__ = [
    "BENCHMARK_ISLAND",
    "EQUITIES_PARQUET",
    "EQUITIES_UNIVERSE_JSON",
    "FUTURES_META_JSON",
    "FUTURES_PARQUET",
    "ISLAND_STYLES",
    "N_STYLES",
    "NUM_ISLANDS",
    "STYLE_NAMES",
    "StyleVector",
    "align_to_sessions",
    "apply_delisting",
    "fetch_yfinance",
    "flag_roll_artifacts",
    "materialize_equities",
    "materialize_futures",
    "process_equities_frame",
    "select_universe",
]


def __getattr__(name: str):
    if name in {
        "EQUITIES_PARQUET", "EQUITIES_UNIVERSE_JSON",
        "apply_delisting", "materialize_equities",
        "process_equities_frame", "select_universe",
    }:
        from src.data.equities import (
            EQUITIES_PARQUET as _EQ_P,
            EQUITIES_UNIVERSE_JSON as _EQ_U,
            apply_delisting as _apply_delisting,
            materialize_equities as _materialize_equities,
            process_equities_frame as _process_equities_frame,
            select_universe as _select_universe,
        )
        return {
            "EQUITIES_PARQUET": _EQ_P,
            "EQUITIES_UNIVERSE_JSON": _EQ_U,
            "apply_delisting": _apply_delisting,
            "materialize_equities": _materialize_equities,
            "process_equities_frame": _process_equities_frame,
            "select_universe": _select_universe,
        }[name]
    if name in {
        "FUTURES_META_JSON", "FUTURES_PARQUET",
        "align_to_sessions", "fetch_yfinance",
        "flag_roll_artifacts", "materialize_futures",
    }:
        from src.data.futures import (
            FUTURES_META_JSON as _FX_M,
            FUTURES_PARQUET as _FX_P,
            align_to_sessions as _align_to_sessions,
            fetch_yfinance as _fetch_yfinance,
            flag_roll_artifacts as _flag_roll_artifacts,
            materialize_futures as _materialize_futures,
        )
        return {
            "FUTURES_META_JSON": _FX_M,
            "FUTURES_PARQUET": _FX_P,
            "align_to_sessions": _align_to_sessions,
            "fetch_yfinance": _fetch_yfinance,
            "flag_roll_artifacts": _flag_roll_artifacts,
            "materialize_futures": _materialize_futures,
        }[name]
    raise AttributeError(f"module 'src.data' has no attribute {name!r}")
