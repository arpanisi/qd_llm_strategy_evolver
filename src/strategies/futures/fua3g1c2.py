import pandas as pd

VOL_LOOKBACK = 20
VOL_SPIKE = 1.5
SPIKE_WEIGHT_MULT = 2.0

def initialize(context):
    context.assets = [future_symbol("ES"), future_symbol("NQ")]
    context.multipliers = {future_symbol("ES"): 50.0, future_symbol("NQ"): 20.0}
    context.base_weight = 1.0 / len(context.assets)

def handle_data(context, data):
    raw_weights = {}
    
    for asset in context.assets:
        volume = data.current(asset, "volume")
        close_price = data.current(asset, "close")
        open_price = data.current(asset, "open")
        
        hist_vol = safe_history(asset, VOL_LOOKBACK + 1, "1d", field="volume")
        avg_vol = hist_vol.mean() if len(hist_vol) >= VOL_LOOKBACK else 0
        
        volume_spike = (avg_vol > 0) and (volume > VOL_SPIKE * avg_vol)
        positive_momentum = (close_price > open_price) if (close_price and open_price) else False
        
        if volume_spike and positive_momentum:
            raw_weights[asset] = SPIKE_WEIGHT_MULT * context.base_weight
        else:
            raw_weights[asset] = context.base_weight
    
    total_weight = sum(raw_weights.values())
    if total_weight == 0:
        return
    
    portfolio_value = context.portfolio.portfolio_value
    
    for asset in context.assets:
        current_weight = raw_weights[asset] / total_weight
        price = data.current(asset, "close")
        multiplier = context.multipliers[asset]
        
        if price and price > 0:
            target_value = current_weight * portfolio_value
            target_contracts = int(round(target_value / (price * multiplier)))
            order(asset, target_contracts)