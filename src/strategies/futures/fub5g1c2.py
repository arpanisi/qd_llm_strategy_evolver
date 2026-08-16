import numpy as np

INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}
RATIO_LOOKBACK = 61
VOL_LOOKBACK = 20
LEG_WEIGHT = 0.5
BASE_Z = 2.0
VOL_ADJUSTMENT = 1.5

def initialize(context):
    context.instruments = {future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()}
    context.prev_position = None

def handle_data(context, data):
    es = next(a for a in context.instruments if a.symbol == "ES")
    nq = next(a for a in context.instruments if a.symbol == "NQ")
    
    pe = safe_history(es, RATIO_LOOKBACK, "1d")
    pn = safe_history(nq, RATIO_LOOKBACK, "1d")
    
    targets = {}
    if len(pe) >= RATIO_LOOKBACK and len(pn) >= RATIO_LOOKBACK:
        ratio_hist = pe / pn
        mu = ratio_hist.iloc[:-1].mean()
        sd = ratio_hist.iloc[:-1].std()
        ratio = ratio_hist.iloc[-1] if pn.iloc[-1] != 0 else 0.0
        
        if sd > 0:
            vol_ratio = ratio_hist.pct_change().dropna()
            if len(vol_ratio) >= VOL_LOOKBACK:
                recent_vol = vol_ratio.iloc[-VOL_LOOKBACK:].std()
                long_term_vol = vol_ratio.std()
                if long_term_vol > 0:
                    vol_factor = recent_vol / long_term_vol
                    dynamic_z = BASE_Z * (1 + VOL_ADJUSTMENT * (vol_factor - 1))
                    dynamic_z = max(1.0, min(3.5, dynamic_z))
                else:
                    dynamic_z = BASE_Z
            else:
                dynamic_z = BASE_Z
            
            if ratio > mu + dynamic_z * sd:
                targets = {nq: LEG_WEIGHT, es: -LEG_WEIGHT}
            elif ratio < mu - dynamic_z * sd:
                targets = {es: LEG_WEIGHT, nq: -LEG_WEIGHT}
    
    pv = context.portfolio.portfolio_value
    for asset, multiplier in context.instruments.items():
        price = data.current(asset, "close")
        if price and price > 0:
            w = targets.get(asset, 0.0)
            contracts = int(round(w * pv / (price * multiplier)))
            order(asset, contracts - context.portfolio.positions[asset].amount)