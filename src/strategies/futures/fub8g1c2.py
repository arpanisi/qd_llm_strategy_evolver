import pandas as pd

INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}

def initialize(context):
    context.assets = {sym: future_symbol(sym) for sym in INSTRUMENTS}
    context.multipliers = INSTRUMENTS
    context.lookback = 60
    context.rebalance_month = None
    
    schedule_function(
        rebalance,
        date_rule=date_rules.every_day(),
        time_rule=time_rules.market_open()
    )

def rebalance(context, data):
    current_date = get_datetime().date()
    
    if context.rebalance_month == current_date.month:
        return
    if current_date.month not in [1, 4, 7, 10]:
        return
    
    context.rebalance_month = current_date.month
    
    returns = {}
    portfolio_value = context.portfolio.portfolio_value
    
    for sym, asset in context.assets.items():
        if not data.can_trade(asset):
            continue
            
        hist = safe_history(asset, context.lookback + 1, "1d")
        if len(hist) < 2:
            continue
            
        price = data.current(asset, "close")
        if pd.isna(price) or price <= 0:
            continue
            
        start_price = hist.iloc[0]
        if start_price <= 0:
            continue
            
        returns[sym] = (price - start_price) / start_price
    
    if len(returns) != 2:
        return
    
    es_return = returns.get("ES", 0)
    nq_return = returns.get("NQ", 0)
    
    total_return = es_return + nq_return
    if abs(total_return) < 1e-8:
        weights = {"ES": 0.5, "NQ": 0.5}
    else:
        weights = {
            "ES": es_return / total_return,
            "NQ": nq_return / total_return
        }
    
    for sym, asset in context.assets.items():
        if not data.can_trade(asset):
            continue
            
        price = data.current(asset, "close")
        if pd.isna(price) or price <= 0:
            continue
            
        target_weight = weights.get(sym, 0)
        multiplier = context.multipliers[sym]
        
        target_notional = portfolio_value * target_weight
        contracts = int(round(target_notional / (price * multiplier)))
        
        current_position = context.portfolio.positions[asset].amount
        if contracts != current_position:
            order(asset, contracts - current_position)