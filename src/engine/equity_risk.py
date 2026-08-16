"""Equity risk enforcement domain (gross leverage cap)."""

from __future__ import annotations

import logging

import zipline.api as zapi
from zipline.assets import Future

from src.config.settings import TrackConfig
from src.engine.context import _CTX

log = logging.getLogger("strategy_runtime")


def _current_gross_exposure(ctx) -> float:
    return float(
        sum(abs(pos.amount * pos.last_sale_price) for pos in ctx.portfolio.positions.values())
    )


def _log_rejection(order_fn: str, asset, reason: str) -> None:
    entry = {"order_fn": order_fn, "asset": str(getattr(asset, "symbol", asset)), "reason": reason}
    _CTX.setdefault("rejections", []).append(entry)
    log.warning("REJECTED %s %s: %s", order_fn, entry["asset"], reason)


def _track() -> TrackConfig:
    return _CTX["track"]


def _equity_projected(context, data) -> tuple[dict, float]:
    """Projected equity book: current filled positions plus this bar's queued
    (not-yet-filled) orders. All same-bar orders fill at bar close, so this is
    the book the cap must hold at the only moment it is economically real."""
    projected: dict = {}
    for a, pos in context.portfolio.positions.items():
        if isinstance(a, Future):
            continue
        projected[a] = float(pos.amount)
    for a, orders in zapi.get_open_orders().items():
        if isinstance(a, Future):
            continue
        for o in orders:
            if o.amount is not None:
                projected[a] = projected.get(a, 0.0) + float(o.amount)
    gross = 0.0
    for a, amount in projected.items():
        if amount:
            gross += abs(amount * float(data.current(a, "close")))
    return projected, gross


def _enforce_equity_cap(context, data) -> None:
    """Cancel same-bar equity orders until the projected end-of-bar gross book
    stays within the gross-leverage cap.

    Order-by-order checks at placement time are inherently wrong: a strategy's
    loop may place buys before the exits that fund them (or vice versa), and
    those transient mid-loop books never fill. Enforcement at bar close is
    order-independent and never false-rejects.
    """
    cap = _track().cost.gross_leverage_cap
    while True:
        projected, gross = _equity_projected(context, data)
        pv = float(context.portfolio.portfolio_value)
        if gross <= cap * pv:
            return
        candidates = []
        for a, orders in zapi.get_open_orders().items():
            if isinstance(a, Future):
                continue
            price = float(data.current(a, "close"))
            for o in orders:
                if o.amount is None:
                    continue
                without = projected[a] - float(o.amount)
                contrib = abs(projected[a] * price) - abs(without * price)
                candidates.append((contrib, o, a))
        candidates.sort(key=lambda c: -c[0])
        best = candidates[0]
        if best[0] <= 0:
            log.warning(
                "gross leverage cap %s exceeded at bar close (projected %.0f > %.0f); "
                "no order cancellation can restore it", cap, gross, cap * pv,
            )
            return
        zapi.cancel_order(best[1].id)
        _log_rejection(
            "order",
            best[2],
            f"gross leverage cap would be exceeded at bar close "
            f"(projected {gross:.0f} > {cap * pv:.0f}); order cancelled",
        )
