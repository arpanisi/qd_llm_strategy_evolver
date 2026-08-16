import pandas as pd

INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}
WARMUP = 91
THRESHOLD = 0.05

def initialize(context):
    context.instruments = {future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()}
    context.threshold = THRESHOLD
    schedule_function(rebalance, date_rules.month_start(), time_rules.market_open())

def rebalance(context, data):
    rets = {}
    for asset in context.instruments:
        prices = safe_history(asset, WARMUP, "1d")
        if len(prices) >= WARMUP:
            rets[asset] = prices.iloc[-1] / prices.iloc[0] - 1.0
    
    best_asset = None
    if rets:
        max_ret = max(rets.values())
        if max_ret > context.threshold:
            for asset, ret in rets.items():
                if ret == max_ret:
                    best_asset = asset
                    break
    
    pv = context.portfolio.portfolio_value
    for asset, multiplier in context.instruments.items():
        if asset == best_asset:
            price = data.current(asset, "close")
            if price and price > 0:
                contracts = int(round(pv / (price * multiplier)))
            else:
                contracts = 0
        else:
            contracts = 0
        order(asset, contracts - context.portfolio.positions[asset].amount)