"""Seed 8 / Island 8 (Benchmark, futures variant): equal-notional buy-and-hold.

Buy equal notional weight of ES and NQ once on the first bar (whole
contracts) and hold with no rebalancing.
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
        contracts = int(round((pv / len(context.instruments)) / (price * multiplier)))
        order_target_contracts(asset, contracts)
