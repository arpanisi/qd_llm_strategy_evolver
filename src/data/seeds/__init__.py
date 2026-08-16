"""Seed strategies (Step 1) for both tracks, plus the calendar helper the
calendar/seasonal seed relies on.

Each island is initialized with a deterministic, self-contained seed strategy:
island `i` uses `seed_<i>_<style>.py` (0..7 = the 8 taxonomy categories,
8 = benchmark). The runner backtests all 9 of a track's seeds over that
track's training window (Step 1 acceptance).

`trading_days_per_month` is computed offline from the track's trading
calendar and injected into the strategy namespace. It is purely
calendar-derived (price-independent), so a seed can identify a month's
first 3 / last 2 trading days without lookahead.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from exchange_calendars import get_calendar

from src.config.settings import TrackConfig

SEEDS_DIR = Path(__file__).resolve().parent


def load_seeds(track: TrackConfig) -> dict[str, str]:
    """stem -> source code for every seed strategy of a track (sorted)."""
    base = SEEDS_DIR / track.name
    return {p.stem: p.read_text() for p in sorted(base.glob("seed_*.py"))}


def trading_days_per_month(track: TrackConfig) -> dict[tuple[int, int], int]:
    """(year, month) -> number of trading sessions in that calendar month.

    Computed over the track's full data range (universe_date .. data_end for
    equities; train start .. test end for futures).
    """
    cal = get_calendar(track.calendar)
    start = pd.Timestamp(track.universe_date or track.train.start)
    end = pd.Timestamp(track.data_end or track.test.end)
    sessions = cal.sessions_in_range(start, end)
    out: dict[tuple[int, int], int] = {}
    for ts in sessions:
        key = (ts.year, ts.month)
        out[key] = out.get(key, 0) + 1
    return out
