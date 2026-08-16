def initialize(context):
    context.assets = [future_symbol("ES"), future_symbol("NQ")]
    context.multipliers = {context.assets[0]: 50.0, context.assets[1]: 20.0}
    context.lookback = 90
    context.warmup = context.lookback + 1
    
    schedule_function(rebalance, date_rules.month_start(), time_rules.market_open())

def rebalance(context, data):
    rets = {}
    
    for asset in context.assets:
        prices = safe_history(asset, context.warmup, "1d")
        if len(prices) >= context.warmup:
            rets[asset] = prices.iloc[-1] / prices.iloc[0] - 1.0
    
    if len(rets) != 2:
        for asset in context.assets:
            order(asset, 0)
        return
    
    es_ret = rets[context.assets[0]]
    nq_ret = rets[context.assets[1]]
    
    if es_ret > 0 and nq_ret > 0:
        if es_ret > nq_ret:
            long_asset = context.assets[0]
            short_asset = context.assets[1]
        else:
            long_asset = context.assets[1]
            short_asset = context.assets[0]
        
        pv = context.portfolio.portfolio_value
        long_price = data.current(long_asset, "close")
        short_price = data.current(short_asset, "close")
        
        if long_price and short_price:
            long_contracts = int(round(pv / (long_price * context.multipliers[long_asset])))
            short_contracts = int(round(pv / (short_price * context.multipliers[short_asset])))
            
            order(long_asset, long_contracts)
            order(short_asset, -short_contracts)
        else:
            for asset in context.assets:
                order(asset, 0)
    else:
        for asset in context.assets:
            order(asset, 0)