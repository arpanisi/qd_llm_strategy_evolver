def initialize(context):
    context.es = future_symbol("ES")
    context.nq = future_symbol("NQ")
    context.multipliers = {context.es: 50.0, context.nq: 20.0}
    
    context.ratio_lookback = 61
    context.vol_lookback = 20
    context.z_min = 1.5
    context.z_max = 2.5
    context.leg_weight = 0.5
    
    context.prev_ratio = None
    context.prev_vol = None

def handle_data(context, data):
    es_hist = safe_history(context.es, context.ratio_lookback, "1d")
    nq_hist = safe_history(context.nq, context.ratio_lookback, "1d")
    
    if len(es_hist) < context.ratio_lookback or len(nq_hist) < context.ratio_lookback:
        return
    
    ratio_series = es_hist / nq_hist
    current_ratio = ratio_series.iloc[-1]
    
    if len(ratio_series) < 2:
        return
    
    mu = ratio_series.iloc[:-1].mean()
    sd = ratio_series.iloc[:-1].std()
    
    if sd <= 0:
        return
    
    vol_series = ratio_series.pct_change().abs().rolling(context.vol_lookback).std()
    current_vol = vol_series.iloc[-1] if not pd.isna(vol_series.iloc[-1]) else context.prev_vol
    
    if current_vol is None:
        context.prev_vol = current_vol
        return
    
    if context.prev_vol is not None and context.prev_vol > 0:
        vol_ratio = current_vol / context.prev_vol
        z_threshold = context.z_max - (context.z_max - context.z_min) * min(vol_ratio, 1.0)
    else:
        z_threshold = context.z_max
    
    context.prev_vol = current_vol
    
    targets = {}
    z_score = (current_ratio - mu) / sd
    
    if z_score > z_threshold:
        targets = {context.nq: context.leg_weight, context.es: -context.leg_weight}
    elif z_score < -z_threshold:
        targets = {context.es: context.leg_weight, context.nq: -context.leg_weight}
    
    pv = context.portfolio.portfolio_value
    for asset in [context.es, context.nq]:
        price = data.current(asset, "close")
        if price is None or price <= 0:
            continue
        
        w = targets.get(asset, 0.0)
        contracts = int(round(w * pv / (price * context.multipliers[asset])))
        
        current_pos = context.portfolio.positions[asset].amount
        if contracts != current_pos:
            order(asset, contracts - current_pos)