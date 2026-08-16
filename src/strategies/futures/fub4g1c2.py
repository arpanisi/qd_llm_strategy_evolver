def initialize(context):
    context.assets = [future_symbol("ES"), future_symbol("NQ")]
    context.multipliers = {future_symbol("ES"): 50.0, future_symbol("NQ"): 20.0}
    context.long = set()
    context.warmup = 21
    context.atr_window = 20
    context.threshold_multiplier = 0.5

def handle_data(context, data):
    longs = set()
    
    for asset in context.assets:
        prices = safe_history(asset, context.warmup, "1d")
        if len(prices) < context.warmup:
            if asset in context.long:
                longs.add(asset)
            continue
        
        prior = prices.iloc[:-1]
        current_close = prices.iloc[-1]
        
        high_20 = prior.max()
        low_20 = prior.min()
        
        if len(prices) >= context.atr_window + 1:
            highs = prices.iloc[-context.atr_window-1:-1]
            lows = prices.iloc[-context.atr_window-1:-1]
            prev_closes = prices.iloc[-context.atr_window-2:-2] if len(prices) >= context.atr_window + 2 else prices.iloc[-context.atr_window-1:-1]
            
            tr1 = highs - lows
            tr2 = abs(highs - prev_closes)
            tr3 = abs(lows - prev_closes)
            true_ranges = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_ranges.mean()
        else:
            atr = 0.0
        
        threshold = context.threshold_multiplier * atr if atr > 0 else 0.0
        
        if current_close > high_20 + threshold:
            longs.add(asset)
        elif current_close < low_20:
            continue
        elif asset in context.long:
            longs.add(asset)
    
    context.long = longs
    
    n_active = len(longs)
    portfolio_value = context.portfolio.portfolio_value
    
    for asset in context.assets:
        multiplier = context.multipliers[asset]
        
        current_price = data.current(asset, "close")
        if current_price is None or current_price <= 0:
            continue
        
        if asset in longs and n_active > 0:
            target_notional = portfolio_value / n_active
            contracts = int(round(target_notional / (current_price * multiplier)))
        else:
            contracts = 0
        
        current_position = context.portfolio.positions[asset].amount
        if contracts != current_position:
            order(asset, contracts - current_position)