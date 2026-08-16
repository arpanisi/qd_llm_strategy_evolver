# Data Schema Specification (generated once; reused by every downstream role)

This document is the single source of truth for what data the strategy engine
and all LLM roles can rely on. Strategy logic must never silently assume data
that is not listed here.

## Shared conventions

- All dates are naive calendar dates (no timezone), stored as `date` (parquet
  `timestamp[ns]`, wall-clock midnight UTC).
- No lookahead: strategy logic may only reference rows timestamped at or
  before the current bar. The engine's data-access interface structurally
  cannot return future bars.
- Returns are total returns where available.

---

## Equities Track (`data/raw/crsp_equities.parquet`)

Source: WRDS `crsp.dsf_v2` (daily stock file, point-in-time), materialized
locally once. **Never queried live from inside the backtest loop.**

Universe: exactly the 15 largest-market-cap CRSP common stocks (share codes
10/11) with continuous coverage, selected as of the first trading day on/after
2015-01-01. See `data/raw/equities_universe.json`.

Date range: 2015-01-02 through 2024-12-31. Splits: train 2015–2020,
validation 2021–2022, test 2023–2024 (fixed).

| Column | Type | Meaning |
|---|---|---|
| `permno` | int | CRSP permanent identifier |
| `dlycaldt` | date | Trading day |
| `ticker` | str | CRSP ticker as of that date |
| `primaryexch` | str | Primary exchange |
| `open` / `high` / `low` / `close` | float | **Total-return-adjusted** OHLC (splits + dividends; adjusted close reproduces CRSP `dlyret`) |
| `volume` | float | Shares traded (unadjusted) |
| `cap` | float | Daily market capitalization, dollars |
| `ret` | float | CRSP `dlyret` total daily return |
| `reti` | float | CRSP `dlyreti` income (dividend) return |
| `split_factor` | float | CRSP `dlycumfacpr` split-only cumulative factor |
| `total_factor` | float | Total-return factor applied to OHLC |
| `delist_flag` | str | CRSP `dlydelflg` |

Delisting: final delisting returns from `crsp.msedelist` are applied on the
delisting date, after which the asset is removed from the tradable universe.

**Not available:** fundamentals, intraday/tape data, options data,
alternative data, index constituents, short-borrow data. Any strategy
requiring these is invalid.

---

## Futures Track (`data/raw/futures.parquet`)

Source: Yahoo Finance continuous front-month futures, `ES=F` (S&P 500
E-mini) and `NQ=F` (Nasdaq-100 E-mini), free, no key.

Universe: exactly `{ES, NQ}`. Point values: ES = $50/point, NQ = $20/point.
Tick size: 0.25 index points (both).

Date range: 2018-01-02 through 2025-12-30. Splits: train 2018–2022,
validation 2023–2024H1, test 2024H2–2025 (fixed, independent of Equities).

| Column | Type | Meaning |
|---|---|---|
| `symbol` | str | `ES` or `NQ` |
| `date` | date | Session date (aligned to the `us_futures` calendar; non-trading sessions forward-fill the prior close) |
| `open` / `high` / `low` / `close` | float | OHLC in index points |
| `volume` | int | Contracts traded |
| `roll_flagged` | bool | Single-day |return| > 8% (candidate roll artifact) — flagged for manual review, never corrected |

**Not available:** true contract-level data, open interest, basis/fair value,
funding or margin schedules, intraday data. The continuous series uses Yahoo's
undocumented front-month splice; roll artifacts are flagged, not repaired.

**Known limitation (accepted):** Yahoo's continuous-contract roll methodology
is undocumented and can splice a single-day jump at a roll date. Flagged days
are logged for review; the engine does not trade on or "fix" them.
