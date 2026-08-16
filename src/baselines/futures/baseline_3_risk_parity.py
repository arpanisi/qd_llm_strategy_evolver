"""Baseline 3 (futures): risk parity (inverse-volatility), rebalanced daily.

Identical formula to the equities version, applied to {ES, NQ}: w_i = (1/sigma_i)
/ sum(1/sigma_j) with sigma_i the rolling 60-day annualized realized volatility.
Target notional w_i x portfolio_value is converted to whole contracts.
Test-window-only baseline (Step 11).
"""

INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}  # symbol -> point value (USD/point)

LOOKBACK = 60
MIN_BARS = 30


def initialize(context):
    context.instruments = {
        future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()
    }


def handle_data(context, data):
    inv = {}
    for asset, multiplier in context.instruments.items():
        prices = data.history(asset, "close", LOOKBACK, "1d")
        if len(prices) < MIN_BARS:
            continue
        vol = prices.pct_change().std() * np.sqrt(BARS_PER_YEAR)
        if vol > 0:
            inv[asset] = 1.0 / vol
    total = sum(inv.values())
    pv = context.portfolio.portfolio_value
    for asset, multiplier in context.instruments.items():
        price = data.current(asset, "close")
        weight = inv.get(asset, 0.0) / total if total > 0 else 0.0
        contracts = int(round(weight * pv / (price * multiplier)))
        order_target_contracts(asset, contracts)
