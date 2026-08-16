import numpy as np

INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}

def initialize(context):
    context.instruments = {future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()}
    context.vol_window = 20
    schedule_function(rebalance, date_rules.week_start(), time_rules.market_open())

def rebalance(context, data):
    if len(context.instruments) == 0:
        return
    
    recent_vols = []
    for asset in context.instruments:
        prices = safe_history(asset, 41, "1d")
        if len(prices) >= 21:
            returns = prices.pct_change().dropna()
            if len(returns) >= 20:
                vol = returns[-20:].std() * np.sqrt(BARS_PER_YEAR)
                if vol > 0:
                    recent_vols.append(vol)
    
    if recent_vols:
        avg_vol = np.mean(recent_vols)
        if avg_vol > 0.25:
            vol_window = 10
        elif avg_vol < 0.15:
            vol_window = 40
        else:
            vol_window = 20
    else:
        vol_window = 20
    
    inv = {}
    for asset in context.instruments:
        prices = safe_history(asset, vol_window + 1, "1d")
        if len(prices) >= vol_window + 1:
            vol = prices.pct_change().std() * np.sqrt(BARS_PER_YEAR)
            if vol > 0:
                inv[asset] = 1.0 / vol
    
    total = sum(inv.values())
    pv = context.portfolio.portfolio_value
    
    for asset, multiplier in context.instruments.items():
        price = data.current(asset, "close")
        if price and total > 0:
            w = inv.get(asset, 0.0) / total
            contracts = int(round(w * pv / (price * multiplier)))
            order(asset, contracts - context.portfolio.positions[asset].amount)
        else:
            order(asset, -context.portfolio.positions[asset].amount)