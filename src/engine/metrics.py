"""Strategy metrics computation (Step 8) and the acceptance criteria evaluation.

Works on a Zipline daily perf DataFrame. One reference implementation shared
by backtests and the standalone path/acceptance tooling so numbers always agree.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

BARS_PER_YEAR = 252


@dataclass
class StrategyMetrics:
    sharpe: float
    sortino: float
    ir: float
    mdd: float
    total_return: float
    trading_frequency: float
    turnover: float
    n_fills: int
    n_days: int
    final_equity: float
    benchmark_ir_base: Optional[float] = None


def _ann(factor: float, bars_per_year: int = BARS_PER_YEAR) -> float:
    return np.sqrt(bars_per_year) * factor


def compute_metrics(
    perf: pd.DataFrame,
    benchmark_returns: Optional[pd.Series] = None,
    bars_per_year: int = BARS_PER_YEAR,
) -> StrategyMetrics:
    """Compute the six core metrics plus trading-frequency/turnover from a
    Zipline daily perf frame.

    `benchmark_returns` is the equal-weighted (baseline-2) daily return series
    aligned to the same sessions as `perf`; when None the IR is reported as
    NaN (baselines vs. themselves).
    """
    rets = perf["returns"].astype(float).fillna(0.0)

    sr = (_ann(rets.mean() / rets.std(ddof=0), bars_per_year)
          if len(rets) > 1 and rets.std(ddof=0) > 0 else np.nan)

    downside = rets[rets < 0]
    sortino = (_ann(rets.mean() / downside.std(ddof=0), bars_per_year)
               if len(downside) > 1 else np.nan)

    if benchmark_returns is not None:
        aligned = benchmark_returns.reindex(rets.index).fillna(0.0)
        active = rets - aligned
        ir = (_ann(active.mean() / active.std(ddof=0), bars_per_year)
              if active.std(ddof=0) > 0 else np.nan)
    else:
        ir = np.nan

    equity = perf["portfolio_value"] if "portfolio_value" in perf else (1 + rets).cumprod() * 1_000_000
    total_return = float((1 + rets).prod() - 1)

    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    mdd = float(((cum - peak) / peak).min()) if len(cum) else np.nan

    n_fills = 0
    if "transactions" in perf:
        for txns in perf["transactions"]:
            if txns is None:
                continue
            try:
                n_fills += len(txns)
            except TypeError:
                pass
    n_days = len(rets)
    trading_frequency = (n_fills * bars_per_year / n_days) if n_days else np.nan

    turnover = _avg_turnover(perf)

    return StrategyMetrics(
        sharpe=float(sr),
        sortino=float(sortino),
        ir=float(ir),
        mdd=float(mdd),
        total_return=float(total_return),
        trading_frequency=float(trading_frequency),
        turnover=float(turnover),
        n_fills=int(n_fills),
        n_days=int(n_days),
        final_equity=float(equity.iloc[-1]) if len(equity) else np.nan,
    )


def _avg_turnover(perf: pd.DataFrame) -> float:
    """Average one-way turnover 0.5 * sum_i |w_{i,t} - w_{i,t-1}| computed
    from per-asset position values / portfolio_value (Step 10), averaged
    over the window's rebalance events (bars with at least one fill)."""
    if "positions" not in perf:
        return np.nan
    weights: list[dict[int, float]] = []
    pvs = perf["portfolio_value"].astype(float) if "portfolio_value" in perf else None
    for i, pos_row in enumerate(perf["positions"]):
        pv = float(pvs.iloc[i]) if pvs is not None else np.nan
        row_w: dict[int, float] = {}
        for pos in pos_row:
            if pos is None:
                continue
            value = pos["amount"] * pos["last_sale_price"]
            row_w[pos["sid"]] = value
        if pv == pv and pv != 0:
            weights.append({k: v / pv for k, v in row_w.items()})
        else:
            weights.append(row_w)
    deltas = []
    for t in range(1, len(weights)):
        if "transactions" in perf:
            txns = perf["transactions"].iloc[t]
            try:
                has_fill = txns is not None and len(txns) > 0
            except TypeError:
                has_fill = False
            if not has_fill:
                continue
        prev, cur = weights[t - 1], weights[t]
        keys = set(prev) | set(cur)
        delta = 0.5 * sum(abs(cur.get(k, 0.0) - prev.get(k, 0.0)) for k in keys)
        if delta > 1e-12:
            deltas.append(delta)
    return float(np.mean(deltas)) if deltas else 0.0


def metrics_to_dict(m: StrategyMetrics) -> dict:
    d = asdict(m)
    return {
        "Sharpe Ratio": d["sharpe"],
        "Sortino Ratio": d["sortino"],
        "Information Ratio": d["ir"],
        "Max Drawdown": d["mdd"],
        "Total Return": d["total_return"],
        "Trading Frequency": d["trading_frequency"],
        "Turnover": d["turnover"],
    }
