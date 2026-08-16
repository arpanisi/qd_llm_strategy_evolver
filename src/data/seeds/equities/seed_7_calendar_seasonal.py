"""Seed 7 / Island 7 (Calendar/Seasonal): turn-of-month effect, daily.

Hold the full equal-weighted portfolio only on each month's first 3 and last
2 trading days; hold cash on all other days. The number of trading days in
each month comes from the injected `trading_days_per_month` calendar schedule
(price-independent, no lookahead).
"""

UNIVERSE = [
    "AAPL", "MSFT", "XOM", "GE", "CVX", "BRK", "PG", "PFE",
    "JNJ", "WFC", "JPM", "WMT", "BAC", "VZ", "ORCL",
]

HEAD = 3
TAIL = 2


def initialize(context):
    context.assets = [symbol(t) for t in UNIVERSE]
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
    weight = 1.0 / len(context.assets) if active else 0.0
    for asset in context.assets:
        order_target_percent(asset, weight)
