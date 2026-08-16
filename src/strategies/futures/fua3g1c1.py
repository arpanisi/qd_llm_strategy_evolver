import numpy as np

INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}
VOL_LOOKBACK = 20
MIN_LOOKBACK = 5
BASE_WEIGHT = 1.0
SCALING_FACTOR = 2.0

def initialize(context):
    context.instruments = {future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()}
    context.prev_weights = {asset: 0.0 for asset in context.instruments}

def handle_data(context, data):
    raw_weights = {}
    
    for asset in context.instruments:
        v = data.current(asset, "volume")
        hist = safe_history(asset, VOL_LOOKBACK + 1, "1d", field="volume")
        
        if len(hist) >= MIN_LOOKBACK:
            avg = hist.mean()
            std = hist.std()
            
            if avg > 0 and std > 0:
                z_score = (v - avg) / std
                weight = BASE_WEIGHT + SCALING_FACTOR * max(0, z_score)
            else:
                weight = BASE_WEIGHT
        else:
            weight = BASE_WEIGHT
        
        raw_weights[asset] = weight
    
    total = sum(raw_weights.values())
    if total > 0:
        pv = context.portfolio.portfolio_value
        
        for asset, multiplier in context.instruments.items():
            price = data.current(asset, "close")
            if price and price > 0:
                target_weight = raw_weights[asset] / total
                target_contracts = int(round((target_weight * pv) / (price * multiplier)))
                
                current_pos = context.portfolio.positions[asset].amount
                if target_contracts != current_pos:
                    order(asset, target_contracts - current_pos)
                    context.prev_weights[asset] = target_weight
            else:
                continue
    else:
        for asset in context.instruments:
            current_pos = context.portfolio.positions[asset].amount
            if current_pos != 0:
                order(asset, -current_pos)
                context.prev_weights[asset] = 0.0