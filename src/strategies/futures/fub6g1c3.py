import numpy as np

INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}
VOL_LOOKBACK = 20
VOL_FLOOR = 0.10

def initialize(context):
    context.instruments = {future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()}
    schedule_function(rebalance, date_rules.week_start(), time_rules.market_open())

def rebalance(context, data):
    inv = {}
    for asset in context.instruments:
        prices = safe_history(asset, VOL_LOOKBACK + 1, "1d")
        if len(prices) >= VOL_LOOKBACK + 1:
            vol = prices.pct_change().std() * np.sqrt(BARS_PER_YEAR)
            vol = max(vol, VOL_FLOOR)
            inv[asset] = 1.0 / vol
    
    total = sum(inv.values())
    pv = context.portfolio.portfolio_value
    
    for asset, multiplier in context.instruments.items():
        price = data.current(asset, "close")
        if price and total > 0:
            w = inv.get(asset, 0.0) / total
            contracts = int(round(w * pv / (price * multiplier)))
            order_target(asset, contracts)
        else:
            order_target(asset, 0)