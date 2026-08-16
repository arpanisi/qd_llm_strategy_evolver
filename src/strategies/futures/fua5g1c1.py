import numpy as np

INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}
RATIO_LOOKBACK = 61
LEG_WEIGHT = 0.5
BASE_Z = 2.0
SHORT_VOL_LOOKBACK = 20
LONG_VOL_LOOKBACK = 200

def initialize(context):
    context.instruments = {future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()}
    context.last_targets = {}

def handle_data(context, data):
    es = next(a for a in context.instruments if a.symbol == "ES")
    nq = next(a for a in context.instruments if a.symbol == "NQ")
    
    pe = safe_history(es, RATIO_LOOKBACK, "1d")
    pn = safe_history(nq, RATIO_LOOKBACK, "1d")
    
    targets = {}
    if len(pe) >= RATIO_LOOKBACK and len(pn) >= RATIO_LOOKBACK:
        ratio_hist = pe / pn
        current_ratio = ratio_hist.iloc[-1]
        
        mu = ratio_hist.iloc[:-1].mean()
        sd = ratio_hist.iloc[:-1].std()
        
        if sd > 0 and len(ratio_hist) >= LONG_VOL_LOOKBACK:
            recent_vol = ratio_hist.iloc[-SHORT_VOL_LOOKBACK:-1].std()
            long_vol = ratio_hist.iloc[-LONG_VOL_LOOKBACK:-1].std()
            
            if long_vol > 0:
                vol_ratio = recent_vol / long_vol
                dynamic_z = BASE_Z * vol_ratio
                dynamic_z = np.clip(dynamic_z, 1.0, 3.0)
                
                if current_ratio > mu + dynamic_z * sd:
                    targets = {nq: LEG_WEIGHT, es: -LEG_WEIGHT}
                elif current_ratio < mu - dynamic_z * sd:
                    targets = {es: LEG_WEIGHT, nq: -LEG_WEIGHT}
            else:
                if current_ratio > mu + BASE_Z * sd:
                    targets = {nq: LEG_WEIGHT, es: -LEG_WEIGHT}
                elif current_ratio < mu - BASE_Z * sd:
                    targets = {es: LEG_WEIGHT, nq: -LEG_WEIGHT}
        elif sd > 0:
            if current_ratio > mu + BASE_Z * sd:
                targets = {nq: LEG_WEIGHT, es: -LEG_WEIGHT}
            elif current_ratio < mu - BASE_Z * sd:
                targets = {es: LEG_WEIGHT, nq: -LEG_WEIGHT}
    
    pv = context.portfolio.portfolio_value
    for asset, multiplier in context.instruments.items():
        price = data.current(asset, "close")
        if price and price > 0:
            w = targets.get(asset, 0.0)
            contracts = int(round(w * pv / (price * multiplier)))
            current_pos = context.portfolio.positions[asset].amount
            if contracts != current_pos:
                order(asset, contracts - current_pos)