def initialize(context):
    context.es = future_symbol("ES")
    context.nq = future_symbol("NQ")
    context.multipliers = {context.es: 50.0, context.nq: 20.0}
    context.month = None
    context.day_in_month = 0
    context.head = 3
    context.tail = 2
    context.lookback = 5

def handle_data(context, data):
    ts = get_datetime()
    month = (ts.year, ts.month)
    
    if month != context.month:
        context.month = month
        context.day_in_month = 0
    context.day_in_month += 1
    
    total = trading_days_per_month.get(month, 21)
    active = context.day_in_month <= context.head or context.day_in_month > total - context.tail
    
    if not active:
        order(context.es, 0)
        order(context.nq, 0)
        return
    
    pv = context.portfolio.portfolio_value
    returns = {}
    
    for asset in [context.es, context.nq]:
        hist = safe_history(asset, context.lookback, "1d")
        if len(hist) >= 2:
            returns[asset] = (hist[-1] / hist[0]) - 1
        else:
            returns[asset] = 0.0
    
    total_ret = abs(returns[context.es]) + abs(returns[context.nq])
    if total_ret > 0:
        es_weight = abs(returns[context.es]) / total_ret
        nq_weight = abs(returns[context.nq]) / total_ret
    else:
        es_weight = nq_weight = 0.5
    
    for asset, weight in [(context.es, es_weight), (context.nq, nq_weight)]:
        price = data.current(asset, "close")
        if price and price > 0:
            contracts = int(round(weight * pv / (price * context.multipliers[asset])))
            order(asset, contracts)