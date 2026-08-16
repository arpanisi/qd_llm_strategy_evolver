"""Futures risk enforcement domain (margin & leverage checks)."""

from __future__ import annotations

from zipline.assets import Future

from src.config.settings import TrackConfig
from src.engine.context import _CTX


def _track() -> TrackConfig:
    return _CTX["track"]


def _futures_price(asset) -> float:
    data = _CTX.get("data")
    ctx = _CTX.get("context")
    if data is not None:
        try:
            return float(data.current(asset, "close"))
        except Exception:
            pass
    pos = ctx.portfolio.positions.get(asset)
    if pos is not None:
        return float(pos.last_sale_price)
    raise ValueError(f"cannot price futures asset {asset}")


def _resulting_contracts(asset, target_contracts) -> dict:
    ctx = _CTX["context"]
    out = {}
    for a, pos in ctx.portfolio.positions.items():
        if not isinstance(a, Future):
            continue
        out[a] = pos.amount
    if target_contracts is None:
        out[asset] = 0.0
    else:
        out[asset] = float(target_contracts)
    return out


def _futures_margin_ok(resulting_contracts: dict, asset, price: float) -> bool:
    """True if required margin of the resulting futures book fits in cash and
    gross notional stays under the leverage cap."""
    ctx = _CTX["context"]
    cost = _track().cost
    result = dict(resulting_contracts)
    total_margin = 0.0
    total_notional = 0.0
    for a, contracts in result.items():
        p = price if a == asset else _futures_price(a)
        total_notional += abs(contracts) * p * a.price_multiplier
        total_margin += abs(contracts) * p * a.price_multiplier * cost.margin_fraction
    pv = float(ctx.portfolio.portfolio_value)
    if total_margin > float(ctx.portfolio.cash):
        return False
    if total_notional > cost.gross_leverage_cap * pv:
        return False
    return True
