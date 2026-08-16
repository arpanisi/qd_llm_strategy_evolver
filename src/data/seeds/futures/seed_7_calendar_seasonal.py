"""Seed 7 / Island 7 (Calendar/Seasonal, futures variant): turn-of-month.

Hold the full equal-weight notional portfolio only on each month's first 3
and last 2 trading days; hold cash otherwise. Trading days per month come
from the injected `trading_days_per_month` calendar schedule. Daily.
"""

INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}  # symbol -> point value (USD/point)

HEAD = 3
TAIL = 2


def initialize(context):
    context.instruments = {
        future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()
    }
    context.month = None
    context.day_in_month = 0


def handle_data(context, data):
    ts = data.current_dt
    month = (ts.year, ts.month)
    if month != context.month:
        context.month = month
        context.day_in_month = 0
    context.day_in_month += 1

    total = trading_days_per_month.get(month, 21)
    active = context.day_in_month <= HEAD or context.day_in_month > total - TAIL
    pv = context.portfolio.portfolio_value
    for asset, multiplier in context.instruments.items():
        price = data.current(asset, "close")
        w = 1.0 / len(context.instruments) if active else 0.0
        contracts = int(round(w * pv / (price * multiplier)))
        order_target_contracts(asset, contracts)
