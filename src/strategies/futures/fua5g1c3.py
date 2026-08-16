import numpy as np

INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}
RATIO_LOOKBACK = 61
Z = 2.0
BASE_WEIGHT = 0.5
VOL_LOOKBACK = 20
TARGET_VOL = 0.15

def initialize(context):
    context.instruments = {future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()}
    context.ratio_history = []
    context.last_targets = {}

def handle_data(context, data):
    es = next(a for a in context.instruments if a.symbol == "ES")
    nq = next(a for a in context.instruments if a.symbol == "NQ")
    
    pe = safe_history(es, RATIO_LOOKBACK, "1d")
    pn = safe_history(nq, RATIO_LOOKBACK, "1d")
    
    targets = {}
    vol_scale = 1.0
    
    if len(pe) >= RATIO_LOOKBACK and len(pn) >= RATIO_LOOKBACK:
        ratio_hist = pe / pn
        mu = ratio_hist.iloc[:-1].mean()
        sd = ratio_hist.iloc[:-1].std()
        ratio = ratio_hist.iloc[-1] if pn.iloc[-1] != 0 else 0.0
        
        if sd > 0:
            if ratio > mu + Z * sd:
                targets = {nq: BASE_WEIGHT, es: -BASE_WEIGHT}
            elif ratio < mu - Z * sd:
                targets = {es: BASE_WEIGHT, nq: -BASE_WEIGHT}
        
        if targets:
            context.ratio_history.append(ratio)
            if len(context.ratio_history) > VOL_LOOKBACK:
                context.ratio_history.pop(0)
            
            if len(context.ratio_history) >= 10:
                returns = np.diff(np.log(context.ratio_history))
                if len(returns) > 0:
                    ann_vol = np.std(returns) * np.sqrt(BARS_PER_YEAR)
                    if ann_vol > 0:
                        vol_scale = min(2.0, TARGET_VOL / ann_vol)
    
    pv = context.portfolio.portfolio_value
    for asset, multiplier in context.instruments.items():
        if not data.can_trade(asset):
            continue
        price = data.current(asset, "close")
        if price <= 0:
            continue
            
        w = targets.get(asset, 0.0) * vol_scale
        contracts = int(round(w * pv / (price * multiplier)))
        
        current_pos = context.portfolio.positions[asset].amount
        if contracts != current_pos:
            order(asset, contracts - current_pos)
    
    context.last_targets = targets