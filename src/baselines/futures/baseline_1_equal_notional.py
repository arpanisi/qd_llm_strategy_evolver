"""Baseline 1 (futures): equal-notional buy-and-hold across ES and NQ.

Test-window-only baseline (Step 11) replacing the equities Market-cap-weighted
baseline, since futures contracts have no market capitalization. At the start
of the test window, size each instrument to equal notional — for budget
B = initial portfolio value, contracts_i = round((B / 2) / (price_i x
point_value_i)) — then hold both positions unchanged with no rebalancing.
"""

INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}  # symbol -> point value (USD/point)


def initialize(context):
    context.instruments = {
        future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()
    }
    context.entered = False


def handle_data(context, data):
    if context.entered:
        return
    context.entered = True
    pv = context.portfolio.portfolio_value
    for asset, multiplier in context.instruments.items():
        price = data.current(asset, "close")
        contracts = int(round((pv / 2.0) / (price * multiplier)))
        order_target_contracts(asset, contracts)
