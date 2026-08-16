import numpy as np

INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}
VOL_WINDOW = 20
REBALANCE_FREQ = 'month_start'

def initialize(context):
    context.instruments = {future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()}
    context.last_rebalance = None
    schedule_function(rebalance, date_rules.month_start(), time_rules.market_open())

def rebalance(context, data):
    momentum_weights = {}
    
    for asset in context.instruments:
        prices = safe_history(asset, VOL_WINDOW * 2 + 1, "1d")
        if len(prices) >= VOL_WINDOW * 2 + 1:
            returns = prices.pct_change().dropna()
            recent_vol = returns[-VOL_WINDOW:].std() * np.sqrt(BARS_PER_YEAR)
            prior_vol = returns[:VOL_WINDOW].std() * np.sqrt(BARS_PER_YEAR)
            
            if prior_vol > 0 and recent_vol > 0:
                vol_momentum = recent_vol / prior_vol
                momentum_weights[asset] = 1.0 / vol_momentum
    
    total = sum(momentum_weights.values())
    pv = context.portfolio.portfolio_value
    
    for asset, multiplier in context.instruments.items():
        price = data.current(asset, "close")
        if price and total > 0:
            weight = momentum_weights.get(asset, 0.0) / total
            target_value = weight * pv
            contracts = int(round(target_value / (price * multiplier)))
            order(asset, contracts - context.portfolio.positions[asset].amount)
        else:
            order(asset, -context.portfolio.positions[asset].amount)