import numpy as np

INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}
VOL_LOOKBACK = 20

def initialize(context):
    context.assets = [future_symbol("ES"), future_symbol("NQ")]
    context.multipliers = {future_symbol("ES"): 50.0, future_symbol("NQ"): 20.0}
    schedule_function(rebalance, date_rules.week_start(), time_rules.market_open())

def rebalance(context, data):
    inv_var = {}
    for asset in context.assets:
        prices = safe_history(asset, VOL_LOOKBACK + 1, "1d")
        if len(prices) >= VOL_LOOKBACK + 1:
            returns = prices.pct_change().dropna()
            if len(returns) > 0:
                vol = returns.std() * np.sqrt(BARS_PER_YEAR)
                if vol > 0:
                    inv_var[asset] = 1.0 / (vol * vol)
    
    total = sum(inv_var.values())
    if total == 0:
        for asset in context.assets:
            order(asset, 0)
        return
    
    pv = context.portfolio.portfolio_value
    
    for asset in context.assets:
        price = data.current(asset, "price")
        if price is None or price <= 0:
            continue
        
        w = inv_var.get(asset, 0.0) / total
        contracts = int(round(w * pv / (price * context.multipliers[asset])))
        current_pos = context.portfolio.positions[asset].amount
        if contracts != current_pos:
            order(asset, contracts - current_pos)