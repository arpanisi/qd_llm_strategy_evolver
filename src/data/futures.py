"""Futures Track data pipeline (Step 1): yfinance ES=F / NQ=F continuous
front-month history, roll-artifact flagging, and local Parquet materialization.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config.settings import RunConfig, TrackConfig

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
FUTURES_PARQUET = DATA_DIR / "futures.parquet"
FUTURES_META_JSON = DATA_DIR / "futures_meta.json"


def fetch_yfinance(
    tickers: list[str], start: str, end: str
) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV for the given tickers, keyed by ticker."""
    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df = df.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["date", "open", "high", "low", "close", "volume"]].dropna(
            subset=["date", "close"]
        )
        out[ticker] = df
    return out


def flag_roll_artifacts(
    df: pd.DataFrame, threshold: float
) -> tuple[pd.DataFrame, list[str]]:
    """Flag single-day returns beyond ``threshold`` (candidate roll artifacts)
    for logged review. Never corrects them."""
    s = df.set_index("date")["close"]
    rets = s.pct_change()
    flagged = rets[rets.abs() > threshold]
    df = df.copy()
    df["roll_flagged"] = df["date"].isin(flagged.index)
    return df, [d.isoformat() for d in flagged.index]


def materialize_futures(cfg: RunConfig) -> dict[str, object]:
    track: TrackConfig = cfg.futures
    raw = fetch_yfinance(
        list(track.yfinance_tickers),
        track.train.start.isoformat(),
        track.test.end.isoformat(),
    )
    sessions = _futures_sessions(track)

    per_ticker: dict[str, dict[str, object]] = {}
    frames: list[pd.DataFrame] = []
    for ticker, df in raw.items():
        sym = track.instruments[track.yfinance_tickers.index(ticker)]
        flagged_df, flagged_dates = flag_roll_artifacts(
            df, float(track.roll_flag_threshold)
        )
        if not flagged_dates:
            flagged_dates = []
        per_ticker[sym] = {
            "ticker": ticker,
            "n_bars": int(len(df)),
            "first_date": df["date"].min().isoformat(),
            "last_date": df["date"].max().isoformat(),
            "flagged_roll_artifact_dates": flagged_dates,
        }
        aligned = align_to_sessions(flagged_df, sessions)
        aligned["symbol"] = sym
        frames.append(aligned)

    combined = pd.concat(frames, ignore_index=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(FUTURES_PARQUET, index=False)
    FUTURES_META_JSON.write_text(
        json.dumps(
            {
                "tracks": per_ticker,
                "point_value": track.point_value,
                "tick_size": track.tick_size,
                "roll_flag_threshold": track.roll_flag_threshold,
            },
            indent=2,
        )
    )
    return {"tickers": per_ticker, "n_total_bars": int(len(combined))}


def align_to_sessions(df: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindex a daily frame onto the futures calendar sessions, forward-filling
    missing sessions (a non-trading holiday still gets the prior close)."""
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    aligned = df.reindex(sessions)
    for col in ("open", "high", "low", "close"):
        aligned[col] = aligned[col].ffill()
    aligned["volume"] = aligned["volume"].fillna(0)
    aligned["roll_flagged"] = aligned["roll_flagged"].fillna(False).astype(bool)
    aligned = aligned.reset_index().rename(columns={"index": "date"})
    return aligned


def _futures_sessions(track: TrackConfig) -> pd.DatetimeIndex:
    from exchange_calendars import get_calendar

    cal = get_calendar(track.calendar)
    return cal.sessions_in_range(track.train.start, track.test.end)
