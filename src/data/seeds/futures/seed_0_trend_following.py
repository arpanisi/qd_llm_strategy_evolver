"""Seed 0 / Island 0 (Trend-Following, futures variant): momentum tilt, monthly.

Each month, go long whichever of ES/NQ has the higher trailing 90-day return
(at full notional); hold cash if both trailing returns are negative.
"""

INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}  # symbol -> point value (USD/point)

WARMUP = 91  # 90 returns + 1 starting close


def initialize(context):
    context.instruments = {
        future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()
    }
    schedule_function(rebalance, date_rules.month_start(), time_rules.market_open())


def rebalance(context, data):
    rets = {}
    for asset in context.instruments:
        prices = safe_history(asset, WARMUP, "1d")
        if len(prices) >= WARMUP:
            rets[asset] = prices.iloc[-1] / prices.iloc[0] - 1.0
    pv = context.portfolio.portfolio_value
    for asset, multiplier in context.instruments.items():
        price = data.current(asset, "close")
        if rets and rets.get(asset) == max(rets.values()) and rets[asset] > 0:
            contracts = int(round(pv / (price * multiplier)))
        else:
            contracts = 0
        order_target_contracts(asset, contracts)
