import numpy as np

INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}
SHORT_LOOKBACK = 5
LONG_LOOKBACK = 60
VOL_LOOKBACK = 20
THRESHOLD_HIGH = 1.2
THRESHOLD_LOW = 0.8
BASE_LEVERAGE = 1.0

def initialize(context):
    context.instruments = {
        future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()
    }
    context.last_weights = {}
    schedule_function(rebalance, date_rules.week_start(), time_rules.market_open())

def rebalance(context, data):
    inv_vol = {}
    vol_ratios = {}
    
    for asset in context.instruments:
        prices_long = safe_history(asset, LONG_LOOKBACK + 1, "1d")
        prices_short = safe_history(asset, SHORT_LOOKBACK + 1, "1d")
        prices_vol = safe_history(asset, VOL_LOOKBACK + 1, "1d")
        
        if len(prices_long) >= LONG_LOOKBACK + 1 and len(prices_short) >= SHORT_LOOKBACK + 1:
            vol_long = prices_long.pct_change().std() * np.sqrt(BARS_PER_YEAR)
            vol_short = prices_short.pct_change().std() * np.sqrt(BARS_PER_YEAR)
            if vol_long > 0:
                vol_ratios[asset] = vol_short / vol_long
        
        if len(prices_vol) >= VOL_LOOKBACK + 1:
            vol = prices_vol.pct_change().std() * np.sqrt(BARS_PER_YEAR)
            if vol > 0:
                inv_vol[asset] = 1.0 / vol
    
    if len(inv_vol) < 2:
        return
    
    total_inv = sum(inv_vol.values())
    base_weights = {asset: inv_vol[asset] / total_inv for asset in inv_vol}
    
    regime_factor = 1.0
    if len(vol_ratios) == 2:
        avg_ratio = np.mean(list(vol_ratios.values()))
        if avg_ratio > THRESHOLD_HIGH:
            regime_factor = 0.7
        elif avg_ratio < THRESHOLD_LOW:
            regime_factor = 1.3
    
    pv = context.portfolio.portfolio_value
    target_value = pv * BASE_LEVERAGE * regime_factor
    
    for asset, multiplier in context.instruments.items():
        price = data.current(asset, "close")
        if price is None or price <= 0:
            continue
            
        w = base_weights.get(asset, 0.0)
        target_contracts = int(round(w * target_value / (price * multiplier)))
        
        current_pos = context.portfolio.positions[asset].amount
        if target_contracts != current_pos:
            order(asset, target_contracts - current_pos)
            context.last_weights[asset] = w
    
    record(regime_factor=regime_factor, avg_vol_ratio=avg_ratio if 'avg_ratio' in locals() else 0.0)