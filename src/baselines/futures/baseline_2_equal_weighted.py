"""Baseline 2 (futures): equal-weighted, rebalanced daily.

w_i = 1/2 per instrument, converted to whole contracts via
round((w_i x portfolio_value) / (price_i x point_value_i)) and rebalanced
daily. This is the Information Ratio benchmark R_b (Step 8A/11) and is
computable over train, validation, and test windows.
"""

INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}  # symbol -> point value (USD/point)


def initialize(context):
    context.instruments = {
        future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()
    }


def handle_data(context, data):
    pv = context.portfolio.portfolio_value
    n = len(context.instruments)
    for asset, multiplier in context.instruments.items():
        price = data.current(asset, "close")
        contracts = int(round((pv / n) / (price * multiplier)))
        order_target_contracts(asset, contracts)
