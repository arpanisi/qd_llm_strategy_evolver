def initialize(context):
    context.es = None
    context.nq = None
    context.multipliers = {}
    
    context.ratio_lookback = 61
    context.z_score = 2.0
    context.leg_weight = 0.5
    context.vol_lookback = 20
    context.vol_threshold = 0.25

def handle_data(context, data):
    if context.es is None or context.nq is None:
        try:
            context.es = symbol("ES")
            context.multipliers[context.es] = 50.0
        except:
            context.es = None
        
        try:
            context.nq = symbol("NQ")
            context.multipliers[context.nq] = 20.0
        except:
            context.nq = None
    
    if context.es is None or context.nq is None:
        return
    
    es_hist = safe_history(context.es, context.ratio_lookback, "1d")
    nq_hist = safe_history(context.nq, context.ratio_lookback, "1d")
    
    if len(es_hist) < context.ratio_lookback or len(nq_hist) < context.ratio_lookback:
        return
    
    ratio_series = es_hist / nq_hist
    mu = ratio_series.iloc[:-1].mean()
    sd = ratio_series.iloc[:-1].std()
    current_ratio = ratio_series.iloc[-1]
    
    if sd <= 0:
        return
    
    es_returns = es_hist.pct_change().dropna()
    nq_returns = nq_hist.pct_change().dropna()
    
    if len(es_returns) >= context.vol_lookback and len(nq_returns) >= context.vol_lookback:
        es_vol = es_returns.iloc[-context.vol_lookback:].std() * np.sqrt(BARS_PER_YEAR)
        nq_vol = nq_returns.iloc[-context.vol_lookback:].std() * np.sqrt(BARS_PER_YEAR)
        avg_vol = (es_vol + nq_vol) / 2
    else:
        avg_vol = 0.0
    
    targets = {}
    if avg_vol < context.vol_threshold:
        if current_ratio > mu + context.z_score * sd:
            targets = {context.nq: context.leg_weight, context.es: -context.leg_weight}
        elif current_ratio < mu - context.z_score * sd:
            targets = {context.es: context.leg_weight, context.nq: -context.leg_weight}
    
    pv = context.portfolio.portfolio_value
    for asset in [context.es, context.nq]:
        price = data.current(asset, "close")
        if price and price > 0:
            weight = targets.get(asset, 0.0)
            contracts = int(round(weight * pv / (price * context.multipliers[asset])))
            order_target(asset, contracts)