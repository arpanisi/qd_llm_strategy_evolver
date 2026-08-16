import numpy as np

INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}
WARMUP = 91
LOOKBACK = 90

def initialize(context):
    context.instruments = {future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()}
    context.prev_weights = {}
    schedule_function(rebalance, date_rules.month_start(), time_rules.market_open())

def rebalance(context, data):
    rets = {}
    for asset in context.instruments:
        prices = safe_history(asset, WARMUP, "1d")
        if len(prices) >= WARMUP:
            rets[asset] = prices.iloc[-1] / prices.iloc[0] - 1.0
    
    if len(rets) == 2:
        assets = list(rets.keys())
        values = np.array([rets[a] for a in assets])
        
        min_val, max_val = values.min(), values.max()
        if max_val > min_val:
            norm_scores = (values - min_val) / (max_val - min_val)
        else:
            norm_scores = np.array([0.5, 0.5])
        
        weighted_avg = np.average(values, weights=norm_scores)
        if weighted_avg > 0:
            weights = norm_scores / norm_scores.sum()
        else:
            weights = np.array([0.0, 0.0])
        
        pv = context.portfolio.portfolio_value
        for i, asset in enumerate(assets):
            multiplier = context.instruments[asset]
            price = data.current(asset, "close")
            if price and price > 0:
                target_value = pv * weights[i]
                contracts = int(round(target_value / (price * multiplier)))
            else:
                contracts = 0
            order(asset, contracts - context.portfolio.positions[asset].amount)
    else:
        for asset in context.instruments:
            order(asset, -context.portfolio.positions[asset].amount)