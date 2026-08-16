def initialize(context):
    context.es = future_symbol("ES")
    context.nq = future_symbol("NQ")
    context.multipliers = {context.es: 50.0, context.nq: 20.0}
    context.month = None
    context.day_in_month = 0
    context.lookback = 5

def handle_data(context, data):
    ts = data.current_dt
    month = (ts.year, ts.month)
    if month != context.month:
        context.month = month
        context.day_in_month = 0
    context.day_in_month += 1
    
    total = trading_days_per_month.get(month, 21)
    active = context.day_in_month <= 3 or context.day_in_month > total - 2
    
    if not active:
        for asset in [context.es, context.nq]:
            order(asset, 0)
        return
    
    pv = context.portfolio.portfolio_value
    returns = {}
    weights = {}
    
    for asset in [context.es, context.nq]:
        hist = safe_history(asset, context.lookback + 1, "1d")
        if len(hist) < 2:
            returns[asset] = 0.0
        else:
            returns[asset] = (hist.iloc[-1] / hist.iloc[0]) - 1
    
    total_momentum = sum(max(0, r) for r in returns.values())
    
    if total_momentum > 0:
        for asset in [context.es, context.nq]:
            weights[asset] = max(0, returns[asset]) / total_momentum
    else:
        for asset in [context.es, context.nq]:
            weights[asset] = 1.0 / len(context.multipliers)
    
    for asset, multiplier in context.multipliers.items():
        price = data.current(asset, "close")
        if price is None:
            continue
        w = weights.get(asset, 0.0)
        contracts = int(round(w * pv / (price * multiplier)))
        order(asset, contracts)