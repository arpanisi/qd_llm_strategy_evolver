"""Equities Track data pipeline (Step 1): WRDS CRSP fetch, universe
selection, delisting handling, and local Parquet materialization.

The backtest engine reads only from the local Parquet file — never from WRDS
live. Credentials come from the .env (WRDS_USERID / WRDS_PGPASS).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import wrds

from src.config.env import wrds_credentials
from src.config.settings import RunConfig, TrackConfig

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
EQUITIES_PARQUET = DATA_DIR / "crsp_equities.parquet"
EQUITIES_UNIVERSE_JSON = DATA_DIR / "equities_universe.json"

DSF_COLUMNS = [
    "permno",
    "dlycaldt",
    "ticker",
    "primaryexch",
    "dlyprc",
    "dlyopen",
    "dlyhigh",
    "dlylow",
    "dlyclose",
    "dlyvol",
    "dlycap",
    "dlyret",
    "dlyreti",
    "dlycumfacpr",
    "dlycumfacshr",
    "dlydelflg",
]

COMMON_SHARE_CODES = (10, 11)

SESSIONS_TOLERANCE_DAYS = 5
COVERAGE_CANDIDATES = 100


def _connect() -> wrds.Connection:
    user, password = wrds_credentials()
    return wrds.Connection(wrds_username=user, wrds_password=password)


def first_trading_day_on_or_after(db: wrds.Connection, date: str) -> pd.Timestamp:
    df = db.raw_sql(
        "SELECT MIN(dlycaldt) AS first_day FROM crsp.dsf_v2 "
        f"WHERE dlycaldt >= '{date}'"
    )
    return pd.Timestamp(df.iloc[0, 0])


def common_stock_permnos_on(db: wrds.Connection, day: str) -> set[int]:
    df = db.raw_sql(
        "SELECT permno FROM crsp.msenames "
        f"WHERE shrcd IN (10, 11) AND namedt <= '{day}' AND nameendt >= '{day}'"
    )
    return set(int(p) for p in df["permno"])


def _top_candidates_on(db: wrds.Connection, day: str, permnos: set[int]) -> pd.DataFrame:
    if not permnos:
        return pd.DataFrame(columns=["permno", "ticker", "dlycap"])
    permno_csv = ",".join(str(p) for p in permnos)
    df = db.raw_sql(
        "SELECT permno, ticker, dlycap FROM crsp.dsf_v2 "
        f"WHERE dlycaldt = '{day}' AND permno IN ({permno_csv}) "
        "AND dlycap IS NOT NULL AND dlycap > 0 "
        "ORDER BY dlycap DESC"
    )
    return df.head(COVERAGE_CANDIDATES)


def _coverage_counts(
    db: wrds.Connection, permnos: set[int], start: str, end: str
) -> dict[int, int]:
    if not permnos:
        return {}
    permno_csv = ",".join(str(p) for p in permnos)
    df = db.raw_sql(
        "SELECT permno, COUNT(DISTINCT dlycaldt) AS n FROM crsp.dsf_v2 "
        f"WHERE dlycaldt >= '{start}' AND dlycaldt <= '{end}' "
        f"AND permno IN ({permno_csv}) GROUP BY permno"
    )
    return {int(r.permno): int(r.n) for r in df.itertuples(index=False)}


def select_universe(
    db: wrds.Connection,
    cfg: TrackConfig,
    sessions: pd.DatetimeIndex,
) -> list[dict[str, object]]:
    """Select the ``n_names`` largest-cap common stocks (share codes 10/11) on
    the first trading day on/after the configured universe date that have
    continuous coverage through ``data_end``."""
    first_day = first_trading_day_on_or_after(db, str(cfg.universe_date))
    common = common_stock_permnos_on(db, first_day.date().isoformat())
    candidates = _top_candidates_on(db, first_day.date().isoformat(), common)
    candidate_permnos = set(int(p) for p in candidates["permno"])

    start = first_day.date().isoformat()
    end = cfg.data_end.isoformat()
    counts = _coverage_counts(db, candidate_permnos, start, end)
    expected = len(sessions)
    tolerance = SESSIONS_TOLERANCE_DAYS

    rows: list[dict[str, object]] = []
    for _, row in candidates.iterrows():
        permno = int(row.permno)
        n = counts.get(permno, 0)
        if expected - n <= tolerance:
            rows.append(
                {
                    "permno": permno,
                    "ticker": row.ticker,
                    "dlycap": float(row.dlycap),
                    "first_day": first_day,
                }
            )
        if len(rows) >= cfg.n_names:
            break
    if len(rows) < cfg.n_names:
        raise RuntimeError(
            f"Only {len(rows)} names passed the continuous-coverage screen; "
            f"needed {cfg.n_names}."
        )
    return rows


def _fetch_history(
    db: wrds.Connection, permnos: list[int], start: str, end: str
) -> pd.DataFrame:
    permno_csv = ",".join(str(p) for p in permnos)
    sql = (
        "SELECT " + ", ".join(DSF_COLUMNS) + " FROM crsp.dsf_v2 "
        f"WHERE dlycaldt >= '{start}' AND dlycaldt <= '{end}' "
        f"AND permno IN ({permno_csv})"
    )
    return db.raw_sql(sql)


def _delisting_returns(
    db: wrds.Connection, permnos: list[int], start: str, end: str
) -> pd.DataFrame:
    permno_csv = ",".join(str(p) for p in permnos)
    df = db.raw_sql(
        "SELECT permno, dlstdt, dlret, dlstcd FROM crsp.msedelist "
        f"WHERE dlstdt >= '{start}' AND dlstdt <= '{end}' AND permno IN ({permno_csv})"
    )
    if df.empty:
        return df
    return df[df["dlret"].notna()]


def process_equities_frame(
    raw: pd.DataFrame, cfg: TrackConfig
) -> pd.DataFrame:
    """Compute total-return-adjusted OHLCV from CRSP fields and coerce types.

    The adjustment is built **per permno**: for each asset, a total-return
    price factor ``TF`` is constructed recursively so that the adjusted-close
    series reproduces ``dlyret`` (total return) exactly:

        TF_0 = 1.0
        TF_t = TF_{t-1} * (1 + R_x_t) / (1 + R_t)

    where ``R_x`` is the split-adjusted price return and ``R`` is ``dlyret``.
    On non-dividend days ``R_x == R`` so ``TF`` equals the split factor; on
    ex-dates the dividend is folded in, and adjusted OHLC stays consistent.
    """
    df = raw.copy().sort_values(["permno", "dlycaldt"]).reset_index(drop=True)
    df["dlycaldt"] = pd.to_datetime(df["dlycaldt"]).dt.tz_localize(None)
    df["dlyret"] = pd.to_numeric(df["dlyret"], errors="coerce")

    for col in ("dlyopen", "dlyhigh", "dlylow", "dlyclose"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["dlyvol"] = pd.to_numeric(df["dlyvol"], errors="coerce").fillna(0)

    factor = pd.Series(np.nan, index=df.index, dtype="float64")
    for permno, grp in df.groupby("permno", sort=False):
        idx = grp.index
        split_factor = grp["dlycumfacpr"].fillna(1.0).replace(0.0, 1.0)
        split_adj_close = grp["dlyclose"] / split_factor
        r_x = split_adj_close.pct_change()
        r = grp["dlyret"]
        ratio = (1.0 + r_x) / (1.0 + r)
        ratio = ratio.where(r.notna(), 1.0).where(r_x.notna(), 1.0)
        ratio = ratio.fillna(1.0).clip(lower=0.0)
        factor.loc[idx] = split_factor * ratio.cumprod()

    for col in ("dlyopen", "dlyhigh", "dlylow", "dlyclose"):
        df[col] = df[col] / factor
    df["dlyprc"] = df["dlyclose"]
    df["dlycumfacshr"] = factor
    return df


def apply_delisting(
    df: pd.DataFrame, delist: pd.DataFrame
) -> tuple[pd.DataFrame, dict[int, float]]:
    """Apply final delisting returns and truncate each affected asset's data
    to its delisting date. Returns (frame, permno->delist_ret)."""
    if delist.empty:
        return df, {}
    applied: dict[int, float] = {}
    for row in delist.itertuples(index=False):
        permno = int(row.permno)
        dlstdt = pd.Timestamp(row.dlstdt).tz_localize(None)
        dlret = float(row.dlret)
        applied[permno] = dlret
        if dlstdt in df.loc[df["permno"] == permno, "dlycaldt"].values:
            mask = (df["permno"] == permno) & (df["dlycaldt"] == dlstdt)
            df.loc[mask, "dlyret"] = dlret
            df = df[~((df["permno"] == permno) & (df["dlycaldt"] > dlstdt))]
        else:
            last = df.loc[df["permno"] == permno, "dlycaldt"].max()
            if pd.notna(last) and dlstdt > last:
                base = df.loc[df["permno"] == permno].iloc[-1].copy()
                base["dlycaldt"] = dlstdt
                base["dlyret"] = dlret
                base["dlyclose"] = base["dlyclose"] * (1 + dlret)
                base["dlyvol"] = 0
                df = pd.concat([df, base.to_frame().T], ignore_index=True)
    return df, applied


def materialize_equities(cfg: RunConfig) -> list[dict[str, object]]:
    """Fetch and materialize the Equities Track dataset. Returns universe rows."""
    track = cfg.equities
    sessions = _xnys_sessions(track)
    db = _connect()
    try:
        universe = select_universe(db, track, sessions)
        permnos = [int(u["permno"]) for u in universe]
        raw = _fetch_history(db, permnos, track.train.start.isoformat(), track.test.end.isoformat())
        delist = _delisting_returns(db, permnos, track.train.start.isoformat(), track.test.end.isoformat())
    finally:
        db.close()

    frame = process_equities_frame(raw, track)
    frame, applied_delist = apply_delisting(frame, delist)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(EQUITIES_PARQUET, index=False)
    universe_meta = [
        {
            "permno": u["permno"],
            "ticker": u["ticker"],
            "dlycap": u["dlycap"],
            "first_day": str(u["first_day"]),
            "delist_ret": applied_delist.get(int(u["permno"])),
        }
        for u in universe
    ]
    EQUITIES_UNIVERSE_JSON.write_text(
        json.dumps({"universe": universe_meta, "sessions_tolerance": SESSIONS_TOLERANCE_DAYS}, indent=2)
    )
    return universe_meta


def _xnys_sessions(track: TrackConfig) -> pd.DatetimeIndex:
    from exchange_calendars import get_calendar

    cal = get_calendar(track.calendar)
    return cal.sessions_in_range(track.universe_date, track.data_end)


def repair_equities_parquet(path: Path = EQUITIES_PARQUET) -> None:
    """Repair a parquet whose OHLC was built with a cross-permno adjustment
    bug: rebuild each asset's price columns from ``dlyret`` (correct CRSP total
    return), preserving intraday OHLC ratios. Anchors adjusted close at 1000 on
    the first session per asset; dollar-% returns are unaffected by the anchor.
    """
    df = pd.read_parquet(path)
    df["dlyret"] = pd.to_numeric(df["dlyret"], errors="coerce")
    buggy_close = df["dlyclose"].astype(float)

    close = pd.Series(np.nan, index=df.index, dtype="float64")
    for permno, grp in df.groupby("permno", sort=False):
        idx = grp.index
        cum = (1.0 + grp["dlyret"].fillna(0.0)).cumprod()
        close.loc[idx] = 1000.0 * cum / cum.iloc[0]

    df["dlyclose"] = close
    for col in ("dlyopen", "dlyhigh", "dlylow"):
        ratio = df[col].astype(float) / buggy_close
        df[col] = close * ratio.where(buggy_close != 0)
        df[col] = df[col].fillna(close)
    df["dlyprc"] = df["dlyclose"]

    # Repair BRK daily volume, which is stored ~1000x too small (e.g. 325
    # shares on 2015-01-02 vs ~3M traded). With VolumeShareSlippage's 2.5%
    # volume limit, the throttled fills make daily-rebalance strategies pile
    # up open orders and blow past the gross-leverage cap. Scaling back up to
    # ~1.9M mean (plausible for BRK.B over 2015-2024) lets normal orders fill.
    brk_mask = df["ticker"] == "BRK"
    df.loc[brk_mask, "dlyvol"] = df.loc[brk_mask, "dlyvol"] * 1000.0

    df.to_parquet(path, index=False)
