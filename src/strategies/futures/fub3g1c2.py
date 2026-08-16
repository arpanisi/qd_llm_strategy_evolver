def initialize(context):
    context.es = future_symbol("ES")
    context.nq = future_symbol("NQ")
    context.multipliers = {context.es: 50.0, context.nq: 20.0}
    context.vol_lookback = 20
    context.low_vol_threshold = 0.75
    context.low_vol_weight_mult = 2.0
    
    context.assets = [context.es, context.nq]
    
    schedule_function(rebalance, date_rules.every_day())

def rebalance(context, data):
    base_weight = 1.0 / len(context.assets)
    raw_weights = {}
    
    for asset in context.assets:
        if not data.can_trade(asset):
            raw_weights[asset] = base_weight
            continue
            
        current_vol = data.current(asset, "volume")
        hist = safe_history(asset, context.vol_lookback + 1, "1d", field="volume")
        
        if len(hist) >= context.vol_lookback and current_vol > 0:
            avg_vol = hist.mean()
            if avg_vol > 0 and current_vol < context.low_vol_threshold * avg_vol:
                raw_weights[asset] = context.low_vol_weight_mult * base_weight
            else:
                raw_weights[asset] = base_weight
        else:
            raw_weights[asset] = base_weight
    
    total = sum(raw_weights.values())
    if total == 0:
        return
    
    pv = context.portfolio.portfolio_value
    
    for asset in context.assets:
        if not data.can_trade(asset):
            continue
            
        price = data.current(asset, "close")
        if price and price > 0:
            target_weight = raw_weights[asset] / total
            multiplier = context.multipliers[asset]
            target_contracts = int(round((target_weight * pv) / (price * multiplier)))
            order(asset, target_contracts)