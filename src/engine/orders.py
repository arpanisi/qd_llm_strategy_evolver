"""Zipline-style order function family with constraint wrappers."""

from __future__ import annotations

from zipline.api import (
    order as _real_order,
    order_percent as _real_order_percent,
    order_target as _real_order_target,
    order_target_percent as _real_order_target_percent,
    order_target_value as _real_order_target_value,
    order_value as _real_order_value,
)
from zipline.assets import Future

from src.engine.context import _CTX
from src.engine.equity_risk import _log_rejection
from src.engine.futures_risk import (
    _futures_margin_ok,
    _futures_price,
    _resulting_contracts,
)


def _position_amount(asset) -> float:
    pos = _CTX["context"].portfolio.positions.get(asset)
    return float(pos.amount) if pos is not None else 0.0


def _safe_order_contracts(asset, contracts, limit_price=None, stop_price=None, style=None):
    if not isinstance(asset, Future):
        return _real_order(asset, contracts, limit_price=limit_price, stop_price=stop_price, style=style)
    resulting = _resulting_contracts(asset, _position_amount(asset) + float(contracts))
    price = _futures_price(asset)
    if _futures_margin_ok(resulting, asset, price):
        return _real_order(asset, contracts, limit_price=limit_price, stop_price=stop_price, style=style)
    _log_rejection("order_contracts", asset, "required margin exceeds available cash or gross cap")
    return None


def _safe_order_target_contracts(asset, target, limit_price=None, stop_price=None, style=None):
    if not isinstance(asset, Future):
        return _real_order_target(asset, target, limit_price=limit_price, stop_price=stop_price, style=style)
    resulting = _resulting_contracts(asset, target)
    price = _futures_price(asset)
    if _futures_margin_ok(resulting, asset, price):
        return _real_order_target(asset, target, limit_price=limit_price, stop_price=stop_price, style=style)
    _log_rejection("order_target_contracts", asset, "required margin exceeds available cash or gross cap")
    return None


def order(asset, amount, limit_price=None, stop_price=None, style=None):
    if isinstance(asset, Future):
        return _safe_order_contracts(asset, amount, limit_price, stop_price, style)
    return _real_order(asset, amount, limit_price=limit_price, stop_price=stop_price, style=style)


def order_value(asset, value, limit_price=None, stop_price=None, style=None):
    if isinstance(asset, Future):
        price = _futures_price(asset)
        contracts = value / (price * asset.price_multiplier)
        return _safe_order_contracts(asset, contracts, limit_price, stop_price, style)
    return _real_order_value(asset, value, limit_price=limit_price, stop_price=stop_price, style=style)


def order_percent(asset, percent, limit_price=None, stop_price=None, style=None):
    if isinstance(asset, Future):
        raise NotImplementedError(
            "order_percent is not supported for the futures track; "
            "specify positions in contracts via order(asset, n_contracts)."
        )
    return _real_order_percent(asset, percent, limit_price=limit_price, stop_price=stop_price, style=style)


def order_target(asset, target, limit_price=None, stop_price=None, style=None):
    if isinstance(asset, Future):
        return _safe_order_target_contracts(asset, target, limit_price, stop_price, style)
    return _real_order_target(asset, target, limit_price=limit_price, stop_price=stop_price, style=style)


def order_target_percent(asset, target, limit_price=None, stop_price=None, style=None):
    if isinstance(asset, Future):
        raise NotImplementedError(
            "order_target_percent is not supported for the futures track; "
            "specify positions in contracts via order(asset, n_contracts)."
        )
    return _real_order_target_percent(asset, target, limit_price=limit_price, stop_price=stop_price, style=style)


def order_target_value(asset, target, limit_price=None, stop_price=None, style=None):
    if isinstance(asset, Future):
        price = _futures_price(asset)
        return _safe_order_target_contracts(
            asset, target / (price * asset.price_multiplier), limit_price, stop_price, style
        )
    return _real_order_target_value(asset, target, limit_price=limit_price, stop_price=stop_price, style=style)
