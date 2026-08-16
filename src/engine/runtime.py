"""Strategy execution engine (Steps 6-8).

Wraps Zipline's run_algorithm with:
  * per-track cost models (Step 7), re-applied AFTER the strategy's own
    initialize() so the standard costs always win;
  * constraint-enforcing order wrappers that shadow the zipline.api order
    functions in the strategy namespace:
      - equities: reject orders that would push gross exposure above the
        200% gross-leverage cap;
      - futures: reject orders that would push required margin (10% of
        resulting notional) above available cash, or gross notional above
        200% of portfolio value;
  * strategy code executed in an isolated namespace (no zipline globals leak).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd
import zipline.api as zapi
from zipline.api import (
    schedule_function as _real_schedule_function,
    set_commission as _real_set_commission,
    set_slippage as _real_set_slippage,
)
from zipline.errors import HistoryWindowStartsBeforeData
from zipline.utils.calendar_utils import get_calendar

from src.config.settings import TrackConfig
from src.engine.costs import (
    equity_commission,
    equity_slippage,
    futures_commission,
    futures_slippage,
)

# Domain splits
from src.engine.context import _CTX
from src.engine.equity_risk import (
    _current_gross_exposure,
    _enforce_equity_cap,
    _equity_projected,
    _log_rejection,
)
from src.engine.futures_risk import (
    _futures_margin_ok,
    _futures_price,
    _resulting_contracts,
)
from src.engine.metrics import StrategyMetrics, compute_metrics
from src.engine.orders import (
    _position_amount,
    _safe_order_contracts,
    _safe_order_target_contracts,
    order,
    order_percent,
    order_target,
    order_target_percent,
    order_target_value,
    order_value,
)

log = logging.getLogger("strategy_runtime")


def schedule_function(func: Callable, *args, **kwargs):
    """Wrap a scheduled callback so the simulation context is available to the
    constraint helpers."""

    def _wrapped_cb(context, data):
        _CTX.update(context=context, data=data)
        func(context, data)
        if _CTX.get("track") is not None and _CTX["track"].name != "futures":
            _enforce_equity_cap(context, data)

    return _real_schedule_function(_wrapped_cb, *args, **kwargs)


def safe_history(asset, bar_count, frequency, field="close"):
    """data.history() that tolerates an insufficient warmup window.

    When the requested window would extend before the bundle's earliest bar,
    Zipline raises HistoryWindowStartsBeforeData. This helper returns an empty
    Series instead so strategies that guard on len() simply sit flat until
    enough history exists (seed strategies run from the first day of the
    training window, where their warmups are not yet available).
    """
    data = _CTX.get("data")
    if data is None:
        return pd.Series(dtype=float)
    try:
        return data.history(asset, field, bar_count, frequency)
    except HistoryWindowStartsBeforeData:
        return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Namespace + execution
# ---------------------------------------------------------------------------

def _build_namespace(track: TrackConfig) -> dict:
    ns: dict = {"np": np, "pd": pd, "math": __import__("math")}
    for name in dir(zapi):
        if not name.startswith("_"):
            ns[name] = getattr(zapi, name)
    ns.update({
        "order": order,
        "order_value": order_value,
        "order_percent": order_percent,
        "order_target": order_target,
        "order_target_percent": order_target_percent,
        "order_target_value": order_target_value,
        "order_contracts": _safe_order_contracts,
        "order_target_contracts": _safe_order_target_contracts,
        "schedule_function": schedule_function,
        "safe_history": safe_history,
        "BARS_PER_YEAR": track.bars_per_year,
    })
    ns["_TRACK"] = track.name
    return ns


def run_strategy(
    source_code: str,
    track: TrackConfig,
    start: str,
    end: str,
    capital_base: float,
    benchmark_returns: Optional[pd.Series] = None,
    extra: Optional[dict] = None,
) -> tuple[pd.DataFrame, StrategyMetrics, dict]:
    """Execute a strategy over [start, end] and return (perf, metrics, info).

    `extra` injects additional read-only names into the strategy namespace
    (e.g. Step 11's market-cap series), provided by the runner, not the code.
    """
    from src.engine.bundle import register_track_bundle

    register_track_bundle(track)

    _CTX.clear()
    _CTX.update(track=track)

    ns = _build_namespace(track)
    ns.update(extra or {})
    try:
        exec(compile(source_code, "<strategy>", "exec"), ns)
    except Exception as e:
        raise RuntimeError(f"strategy failed to compile: {e}") from e

    user_initialize = ns.get("initialize")
    user_handle_data = ns.get("handle_data")
    user_bts = ns.get("before_trading_start")
    if user_initialize is None:
        raise RuntimeError("strategy has no initialize() function")

    is_futures = track.name == "futures"

    def _apply_standard_costs(context):
        if is_futures:
            _real_set_commission(us_futures=futures_commission(track))
            _real_set_slippage(us_futures=futures_slippage(track))
        else:
            _real_set_commission(us_equities=equity_commission(track))
            _real_set_slippage(us_equities=equity_slippage(track))

    def _wrapped_initialize(context):
        user_initialize(context)
        _apply_standard_costs(context)

    def _wrapped_handle_data(context, data):
        _CTX.update(context=context, data=data, track=track)
        user_handle_data(context, data)
        if not is_futures:
            _enforce_equity_cap(context, data)

    def _wrapped_bts(context, data):
        _CTX.update(context=context, data=data, track=track)
        user_bts(context, data)

    if user_bts is None:
        _wrapped_bts = None

    from zipline.utils.run_algo import run_algorithm

    perf = run_algorithm(
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
        initialize=_wrapped_initialize,
        handle_data=_wrapped_handle_data if user_handle_data else None,
        before_trading_start=_wrapped_bts if user_bts else None,
        capital_base=float(capital_base),
        data_frequency="daily",
        bundle=track.bundle_name,
        trading_calendar=get_calendar(track.calendar),
        metrics_set="default",
        benchmark_returns=None,
    )

    metrics = compute_metrics(perf, benchmark_returns=benchmark_returns, bars_per_year=track.bars_per_year)
    info = {
        "rejections": list(_CTX.get("rejections", [])),
        "n_rejections": len(_CTX.get("rejections", [])),
    }
    return perf, metrics, info


def compute_perf_metrics(perf: pd.DataFrame, benchmark_returns: Optional[pd.Series] = None) -> StrategyMetrics:
    return compute_metrics(perf, benchmark_returns=benchmark_returns)
