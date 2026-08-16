import numpy as np

INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}
VOL_LOOKBACK = 5

def initialize(context):
    context.es = continuous_future('ES', offset=0, roll='calendar')
    context.nq = continuous_future('NQ', offset=0, roll='calendar')
    context.assets = [context.es, context.nq]
    context.multipliers = {context.es: 50.0, context.nq: 20.0}
    schedule_function(rebalance, date_rules.every_day(), time_rules.market_open())

def rebalance(context, data):
    inv_vol = {}
    for asset in context.assets:
        prices = safe_history(asset, VOL_LOOKBACK + 1, "1d")
        if len(prices) >= VOL_LOOKBACK + 1:
            returns = prices.pct_change().dropna()
            if len(returns) >= VOL_LOOKBACK:
                vol = returns.std() * np.sqrt(BARS_PER_YEAR)
                if vol > 0:
                    inv_vol[asset] = 1.0 / vol
    
    if len(inv_vol) < 2:
        return
    
    total = sum(inv_vol.values())
    pv = context.portfolio.portfolio_value
    
    for asset in context.assets:
        price = data.current(asset, "close")
        if price is None or price <= 0:
            continue
            
        weight = inv_vol.get(asset, 0.0) / total
        target_value = weight * pv
        contracts = int(round(target_value / (price * context.multipliers[asset])))
        
        current_pos = 0
        pos_key = asset
        if pos_key in context.portfolio.positions:
            current_pos = context.portfolio.positions[pos_key].amount
        
        if contracts != current_pos:
            order_target(asset, contracts)