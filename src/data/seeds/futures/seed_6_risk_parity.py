"""Seed 6 / Island 6 (Risk Parity, futures variant): inverse-20d-vol, weekly.

Weight each instrument inversely proportional to its trailing 20-day
annualized realized volatility, converted to whole contracts. Same formula as
Seed 2 (per coding-plan.md, risk-allocation == volatility at seed level).
"""

INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}  # symbol -> point value (USD/point)

VOL_LOOKBACK = 20


def initialize(context):
    context.instruments = {
        future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()
    }
    schedule_function(rebalance, date_rules.week_start(), time_rules.market_open())


def rebalance(context, data):
    inv = {}
    for asset in context.instruments:
        prices = safe_history(asset, VOL_LOOKBACK + 1, "1d")
        if len(prices) >= VOL_LOOKBACK + 1:
            vol = prices.pct_change().std() * np.sqrt(BARS_PER_YEAR)
            if vol > 0:
                inv[asset] = 1.0 / vol
    total = sum(inv.values())
    pv = context.portfolio.portfolio_value
    for asset, multiplier in context.instruments.items():
        price = data.current(asset, "close")
        w = inv.get(asset, 0.0) / total if total > 0 else 0.0
        contracts = int(round(w * pv / (price * multiplier)))
        order_target_contracts(asset, contracts)
