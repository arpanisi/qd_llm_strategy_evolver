def initialize(context):
    context.assets = [sid(8554), sid(8460)]
    context.multipliers = {context.assets[0]: 50.0, context.assets[1]: 20.0}
    schedule_function(rebalance, date_rules.month_start(), time_rules.market_open())

def rebalance(context, data):
    ret_90 = {}
    ret_30 = {}
    
    for asset in context.assets:
        prices_90 = safe_history(asset, 91, "1d")
        if len(prices_90) >= 91:
            ret_90[asset] = prices_90.iloc[-1] / prices_90.iloc[0] - 1.0
        
        prices_30 = safe_history(asset, 31, "1d")
        if len(prices_30) >= 31:
            ret_30[asset] = prices_30.iloc[-1] / prices_30.iloc[0] - 1.0
    
    if not ret_90 or not ret_30:
        for asset in context.assets:
            order_target(asset, 0)
        return
    
    best_asset = max(ret_90, key=ret_90.get)
    
    if ret_90[best_asset] > 0 and ret_30.get(best_asset, -1) > 0:
        price = data.current(best_asset, "close")
        multiplier = context.multipliers[best_asset]
        contracts = int(round(context.portfolio.portfolio_value / (price * multiplier)))
        for asset in context.assets:
            if asset == best_asset:
                order_target(asset, contracts)
            else:
                order_target(asset, 0)
    else:
        for asset in context.assets:
            order_target(asset, 0)