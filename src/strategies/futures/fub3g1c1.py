INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}
VOL_LOOKBACK = 20
VOL_SPIKE = 1.5
SPIKE_WEIGHT_MULT = 2.0
MOMENTUM_LOOKBACK = 5

def initialize(context):
    context.instruments = {future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()}
    context.base_weight = 1.0 / len(context.instruments)

def handle_data(context, data):
    raw_weights = {}
    for asset in context.instruments:
        volume = data.current(asset, "volume")
        price = data.current(asset, "close")
        
        vol_hist = safe_history(asset, VOL_LOOKBACK + 1, "1d", field="volume")
        price_hist = safe_history(asset, MOMENTUM_LOOKBACK + 1, "1d", field="close")
        
        vol_avg = vol_hist.mean() if len(vol_hist) >= VOL_LOOKBACK else None
        momentum = (price_hist.iloc[-1] / price_hist.iloc[0] - 1) if len(price_hist) >= MOMENTUM_LOOKBACK + 1 else None
        
        volume_spike = vol_avg and vol_avg > 0 and volume > VOL_SPIKE * vol_avg
        positive_momentum = momentum and momentum > 0
        
        if volume_spike and positive_momentum:
            raw_weights[asset] = SPIKE_WEIGHT_MULT * context.base_weight
        else:
            raw_weights[asset] = context.base_weight
    
    total = sum(raw_weights.values())
    if total == 0:
        return
    
    pv = context.portfolio.portfolio_value
    for asset, multiplier in context.instruments.items():
        price = data.current(asset, "close")
        if price and price > 0:
            target_weight = raw_weights[asset] / total
            target_value = target_weight * pv
            contracts = int(round(target_value / (price * multiplier)))
            order(asset, contracts - context.portfolio.positions[asset].amount)