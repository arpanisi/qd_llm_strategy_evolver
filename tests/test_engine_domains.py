"""Unit tests for extracted engine domain modules (orders, equity_risk, futures_risk)."""

import itertools

import pytest
import pandas as pd
import numpy as np
from zipline.assets import Equity, ExchangeInfo, Future

from src.engine.context import _CTX
from src.engine.equity_risk import _current_gross_exposure, _log_rejection
from src.engine.futures_risk import _futures_margin_ok, _resulting_contracts
from src.engine.orders import _position_amount


class DummyPosition:
    def __init__(self, amount, last_sale_price):
        self.amount = amount
        self.last_sale_price = last_sale_price


class DummyPortfolio:
    def __init__(self, positions, cash=100000.0, portfolio_value=100000.0):
        self.positions = positions
        self.cash = cash
        self.portfolio_value = portfolio_value


class DummyContext:
    def __init__(self, positions, cash=100000.0, portfolio_value=100000.0):
        self.portfolio = DummyPortfolio(positions, cash, portfolio_value)


# Real zipline.assets objects, not a hand-rolled stand-in: Finding 1's root cause
# was exactly a test double (a prior `DummyAsset` with a fabricated `asset_type`
# attribute and a fabricated `.multiplier` attribute) that didn't match what real
# `zipline.assets.Future`/`Equity` objects actually expose, so a test built on it
# could not have caught the bug either. Building fixtures from the real classes
# means these tests fail the same way production would.
_SID_COUNTER = itertools.count(1)
_CME = ExchangeInfo("CME", "CME", "US")
_NYSE = ExchangeInfo("XNYS", "XNYS", "US")


def _make_future(symbol="ES", multiplier=50.0):
    sid = next(_SID_COUNTER)
    return Future(
        sid=sid,
        exchange_info=_CME,
        symbol=symbol,
        root_symbol=symbol,
        asset_name=f"{symbol} Future",
        start_date=pd.Timestamp("2020-01-01"),
        end_date=pd.Timestamp("2025-01-01"),
        first_traded=pd.Timestamp("2020-01-01"),
        notice_date=pd.Timestamp("2025-01-01"),
        expiration_date=pd.Timestamp("2025-01-01"),
        auto_close_date=pd.Timestamp("2025-01-01"),
        tick_size=0.25,
        multiplier=multiplier,
    )


def _make_equity(symbol="AAPL"):
    sid = next(_SID_COUNTER)
    return Equity(sid=sid, exchange_info=_NYSE, symbol=symbol, asset_name=symbol)


class DummyTrackCost:
    def __init__(self, margin_fraction=0.10, gross_leverage_cap=2.0):
        self.margin_fraction = margin_fraction
        self.gross_leverage_cap = gross_leverage_cap


class DummyTrackConfig:
    def __init__(self, name="equities"):
        self.name = name
        self.cost = DummyTrackCost()


def test_equity_risk_exposure_and_log_rejection():
    _CTX.clear()
    a1 = _make_equity("AAPL")
    a2 = _make_equity("MSFT")
    ctx = DummyContext({
        a1: DummyPosition(10, 150.0),
        a2: DummyPosition(-5, 300.0),
    })

    # Gross exposure: 10 * 150 + |-5| * 300 = 1500 + 1500 = 3000
    exposure = _current_gross_exposure(ctx)
    assert exposure == 3000.0

    _log_rejection("order", a1, "test rejection")
    assert len(_CTX["rejections"]) == 1
    assert _CTX["rejections"][0]["asset"] == "AAPL"


def test_futures_risk_resulting_contracts_and_margin():
    _CTX.clear()
    fut = _make_future("ES", multiplier=50.0)
    ctx = DummyContext({fut: DummyPosition(2, 4000.0)}, cash=500000.0, portfolio_value=500000.0)
    _CTX["context"] = ctx
    _CTX["track"] = DummyTrackConfig("futures")

    pos_amt = _position_amount(fut)
    assert pos_amt == 2.0

    res = _resulting_contracts(fut, 4)
    assert res[fut] == 4.0

    # Margin required: 4 contracts * 4000 price * 50 multiplier * 0.10 margin_fraction = 80,000
    # Portfolio cash = 100,000 -> fits!
    assert _futures_margin_ok(res, fut, 4000.0) is True

    # 10 contracts -> margin = 200,000 > cash (100,000) -> fails!
    res_large = _resulting_contracts(fut, 10)
    assert _futures_margin_ok(res_large, fut, 4000.0) is False


def test_resulting_contracts_ignores_real_equity_positions():
    """isinstance(a, Future) must correctly exclude a real Equity position from
    the futures book, exercising the same branch condition Finding 1 fixed."""
    _CTX.clear()
    fut = _make_future("ES", multiplier=50.0)
    eq = _make_equity("AAPL")
    ctx = DummyContext(
        {fut: DummyPosition(2, 4000.0), eq: DummyPosition(100, 150.0)},
        cash=500000.0,
        portfolio_value=500000.0,
    )
    _CTX["context"] = ctx

    res = _resulting_contracts(fut, 4)
    assert set(res) == {fut}
    assert res[fut] == 4.0


def test_repeated_futures_orders_are_capped_by_margin_not_unbounded(monkeypatch):
    """Regression test for Finding 1 (critical): before the fix, every risk
    check branched on `getattr(asset, "asset_type", "") == "future"`, which is
    always False for real `zipline.assets.Future` objects (they carry no
    `asset_type` attribute at all), so `order()` always fell through to the
    raw, unconstrained zipline order call and `_futures_margin_ok` was never
    invoked. A strategy that repeatedly called `order(asset, positive_amount)`
    every bar (absolute order, not order_target) could accumulate an unbounded
    futures position with no cap ever engaging.

    This test calls the real `src.engine.orders.order()` entry point directly,
    against a real `Future` asset, many times in a row with a fixed positive
    size, and asserts the position is capped once required margin would exceed
    available cash -- rather than growing without bound -- and that every
    rejection beyond the cap is recorded via `_log_rejection`.
    """
    import src.engine.orders as orders_mod

    _CTX.clear()
    fut = _make_future("ES", multiplier=50.0)
    # price=100, multiplier=50 -> notional/contract=5,000, margin/contract=500.
    # cash=2,000 -> margin allows exactly 4 contracts (2,000 / 500); the 5th
    # would require 2,500 > 2,000 and must be rejected. portfolio_value is set
    # high enough that the separate gross-leverage-cap check never binds here,
    # isolating the margin-vs-cash check under test.
    position = DummyPosition(0.0, 100.0)
    ctx = DummyContext({fut: position}, cash=2000.0, portfolio_value=100000.0)
    _CTX["context"] = ctx
    _CTX["track"] = DummyTrackConfig("futures")

    filled = []

    def fake_real_order(asset, amount, limit_price=None, stop_price=None, style=None):
        filled.append(amount)
        position.amount += amount
        return "order-id"

    monkeypatch.setattr(orders_mod, "_real_order", fake_real_order)

    for _ in range(10):
        orders_mod.order(fut, 1)

    assert isinstance(fut, Future)
    assert len(filled) == 4, "margin cap should have accepted exactly 4 contracts"
    assert position.amount == 4.0, "position must not grow past the margin-capped size"
    assert len(_CTX["rejections"]) == 6, "the remaining 6 repeated orders must be rejected, not filled"
    assert all(r["order_fn"] == "order_contracts" for r in _CTX["rejections"])
    assert all("margin" in r["reason"] for r in _CTX["rejections"])
