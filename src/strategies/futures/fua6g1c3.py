def initialize(context):
    context.assets = [symbol("ES"), symbol("NQ")]
    context.multipliers = {context.assets[0]: 50.0, context.assets[1]: 20.0}
    context.vol_lookback = 5
    schedule_function(rebalance, date_rules.every_day(), time_rules.market_open())

def rebalance(context, data):
    inv_vol = {}
    for asset in context.assets:
        try:
            prices = safe_history(asset, context.vol_lookback + 1, "1d")
        except:
            continue
        if len(prices) >= context.vol_lookback + 1:
            returns = prices.pct_change().dropna()
            if len(returns) >= context.vol_lookback:
                vol = returns.std() * np.sqrt(BARS_PER_YEAR)
                if vol > 0:
                    inv_vol[asset] = 1.0 / vol
    
    if len(inv_vol) < 2:
        return
    
    total = sum(inv_vol.values())
    pv = context.portfolio.portfolio_value
    
    for asset in context.assets:
        try:
            price = data.current(asset, "close")
        except:
            continue
        if price is None or price <= 0:
            continue
            
        weight = inv_vol.get(asset, 0.0) / total if total > 0 else 0.0
        multiplier = context.multipliers[asset]
        target_contracts = int(round(weight * pv / (price * multiplier)))
        
        current_pos = context.portfolio.positions[asset].amount
        if target_contracts != current_pos:
            order(asset, target_contracts - current_pos)