"""Zipline data bundle ingestion for both tracks (Step 6).

Each track has its own separately-registered bundle. The ingest function
reads the track's local Parquet file (Step 1) and writes Zipline's bcolz
daily-bar structure. Bundles are ingested once and reused by every strategy.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from exchange_calendars import get_calendar
from zipline.data.bundles import core as bundles

from src.config.settings import RunConfig, TrackConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Parquet -> per-asset frames
# ---------------------------------------------------------------------------

def _equities_frames() -> dict[int, pd.DataFrame]:
    df = pd.read_parquet(REPO_ROOT / "data" / "raw" / "crsp_equities.parquet")
    df = df.sort_values(["permno", "dlycaldt"])
    cal = get_calendar("XNYS")
    frames: dict[int, pd.DataFrame] = {}
    ticker_by_permno: dict[int, str] = {}
    for permno, grp in df.groupby("permno"):
        grp = grp.set_index(pd.DatetimeIndex(grp["dlycaldt"]))
        bars = grp[["dlyopen", "dlyhigh", "dlylow", "dlyclose", "dlyvol"]].copy()
        bars.columns = ["open", "high", "low", "close", "volume"]
        bars["volume"] = pd.to_numeric(bars["volume"]).astype("int64")
        sessions = cal.sessions_in_range(bars.index.min(), bars.index.max())
        bars = bars.reindex(sessions).dropna(subset=["close"])
        frames[int(permno)] = bars
        ticker_by_permno[int(permno)] = str(grp["ticker"].iloc[-1])
    return frames, ticker_by_permno


def _futures_frames() -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(REPO_ROOT / "data" / "raw" / "futures.parquet")
    frames: dict[str, pd.DataFrame] = {}
    for symbol, grp in df.groupby("symbol"):
        grp = grp.set_index(pd.DatetimeIndex(grp["date"])).sort_index()
        bars = grp[["open", "high", "low", "close", "volume"]].copy()
        bars["volume"] = pd.to_numeric(bars["volume"]).fillna(0).astype("int64")
        frames[str(symbol)] = bars
    return frames


def equities_sid_map() -> dict[int, int]:
    """permno -> sid, using the same ordering the bundle ingest assigns."""
    frames, _ = _equities_frames()
    return {permno: sid for sid, permno in enumerate(sorted(frames))}


def equities_market_caps() -> dict[int, pd.Series]:
    """sid -> daily CRSP market cap (dlycap) series, indexed by timestamp.

    Used by Step 11's market-cap-weighted baseline; injected into the strategy
    namespace by the runner. Indexed by sid to match how the bundle orders
    assets (sorted permno).
    """
    df = pd.read_parquet(REPO_ROOT / "data" / "raw" / "crsp_equities.parquet")
    sid_of = equities_sid_map()
    out: dict[int, pd.Series] = {}
    for permno, grp in df.groupby("permno", sort=True):
        s = (
            grp.set_index(pd.DatetimeIndex(grp["dlycaldt"]))
            .sort_index()["dlycap"]
            .astype(float)
        )
        out[sid_of[int(permno)]] = s
    return out


# ---------------------------------------------------------------------------
# Ingest functions
# ---------------------------------------------------------------------------

def _equities_ingest(environ, asset_db_writer, minute_bar_writer, daily_bar_writer,
                     adjustment_writer, calendar, start_session, end_session,
                     cache, show_progress, output_dir):
    frames, ticker_by_permno = _equities_frames()
    permnos = sorted(frames.keys())
    dtype = [("start_date", "datetime64[ns]"), ("end_date", "datetime64[ns]"),
             ("auto_close_date", "datetime64[ns]"), ("symbol", "object")]
    metadata = pd.DataFrame(np_empty(len(permnos), dtype))

    tickers = list(ticker_by_permno[p] for p in permnos)
    if len(set(tickers)) != len(tickers):
        raise ValueError("duplicate tickers in equities universe; use permno as symbol")
    sid_map = {permno: sid for sid, permno in enumerate(permnos)}

    for sid, permno in enumerate(permnos):
        bars = frames[permno]
        start = bars.index[0]
        end = bars.index[-1]
        metadata.iloc[sid] = start, end, end + pd.Timedelta(days=1), ticker_by_permno[permno]

    daily_bar_writer.write(
        ((sid, frames[permno]) for permno, sid in sid_map.items()),
        show_progress=show_progress,
    )
    metadata["exchange"] = "XNYS"
    asset_db_writer.write(equities=metadata)
    adjustment_writer.write(
        splits=pd.DataFrame(columns=["sid", "ratio", "effective_date"]),
        dividends=pd.DataFrame(
            columns=["sid", "amount", "ex_date", "record_date", "declared_date", "pay_date"]
        ),
    )


def _futures_ingest(environ, asset_db_writer, minute_bar_writer, daily_bar_writer,
                    adjustment_writer, calendar, start_session, end_session,
                    cache, show_progress, output_dir):
    frames = _futures_frames()
    symbols = sorted(frames.keys())
    point_value = {  # from config; kept minimal here, validated by caller
        "ES": 50.0, "NQ": 20.0,
    }
    tick_size = 0.25

    dtype = [("start_date", "datetime64[ns]"), ("end_date", "datetime64[ns]"),
             ("auto_close_date", "datetime64[ns]"), ("symbol", "object"),
             ("root_symbol", "object"), ("multiplier", "float"), ("tick_size", "float")]
    metadata = pd.DataFrame(np_empty(len(symbols), dtype))
    root_symbols = pd.DataFrame(columns=["root_symbol", "root_symbol_id", "description", "exchange"])

    for sid, symbol in enumerate(symbols):
        bars = frames[symbol]
        start = bars.index[0]
        end = bars.index[-1]
        metadata.iloc[sid] = (
            start, end, end + pd.Timedelta(days=1), symbol, symbol,
            point_value[symbol], tick_size,
        )
        root_symbols.loc[sid] = [symbol, sid, f"{symbol} front-month continuous", "CME"]

    daily_bar_writer.write(
        ((sid, frames[symbol]) for sid, symbol in enumerate(symbols)),
        show_progress=show_progress,
    )
    metadata["exchange"] = "CME"
    asset_db_writer.write(futures=metadata, root_symbols=root_symbols)
    adjustment_writer.write(
        splits=pd.DataFrame(columns=["sid", "ratio", "effective_date"]),
        dividends=pd.DataFrame(
            columns=["sid", "amount", "ex_date", "record_date", "declared_date", "pay_date"]
        ),
    )


def np_empty(size: int, dtype):
    import numpy as np
    return np.empty(size, dtype=dtype)


# ---------------------------------------------------------------------------
# Registration + ingestion
# ---------------------------------------------------------------------------

def _register(name: str, ingest_fn, calendar_name: str, start_session, end_session) -> None:
    if name in bundles.bundles:
        return
    bundles.register(
        name,
        ingest_fn,
        calendar_name=calendar_name,
        start_session=start_session,
        end_session=end_session,
    )


def register_track_bundle(track: TrackConfig) -> None:
    """Register a single track's bundle (idempotent) so run_algorithm can load it."""
    if track.is_futures:
        frames = _futures_frames()
        cal = get_calendar(track.calendar)
        sessions = cal.sessions_in_range(
            min(f.index.min() for f in frames.values()),
            max(f.index.max() for f in frames.values()),
        )
        _register(
            track.bundle_name,
            _futures_ingest,
            track.calendar,
            sessions.min(),
            sessions.max(),
        )
    else:
        frames, _ = _equities_frames()
        cal = get_calendar(track.calendar)
        sessions = cal.sessions_in_range(
            min(f.index.min() for f in frames.values()),
            max(f.index.max() for f in frames.values()),
        )
        _register(
            track.bundle_name,
            _equities_ingest,
            track.calendar,
            sessions.min(),
            sessions.max(),
        )


def register_bundles(cfg: RunConfig) -> None:
    for track in cfg.tracks:
        register_track_bundle(track)


def _bundle_exists(name: str) -> bool:
    from zipline.utils.paths import data_path
    path = data_path([name])
    return Path(path).exists()


def ensure_bundles_ingested(cfg: RunConfig, force: bool = False) -> None:
    register_bundles(cfg)
    for track in cfg.tracks:
        if force or not _bundle_exists(track.bundle_name):
            bundles.ingest(track.bundle_name, show_progress=False)
