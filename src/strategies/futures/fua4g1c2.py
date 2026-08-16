def initialize(context):
    context.assets = [future_symbol('ES'), future_symbol('NQ')]
    context.multipliers = {future_symbol('ES'): 50.0, future_symbol('NQ'): 20.0}
    context.lookback = 10
    context.confirmation_pct = 0.02
    context.long_positions = set()
    context.warmup_needed = context.lookback + 1

def handle_data(context, data):
    current_longs = set()
    
    for asset in context.assets:
        prices = safe_history(asset, context.warmup_needed, "1d")
        if len(prices) < context.warmup_needed:
            continue
            
        current_price = data.current(asset, "close")
        if pd.isna(current_price):
            continue
            
        prior_prices = prices.iloc[:-1]
        if len(prior_prices) < context.lookback:
            continue
            
        high_10d = prior_prices.max()
        low_10d = prior_prices.min()
        
        threshold = high_10d * (1 + context.confirmation_pct)
        
        if current_price > threshold:
            current_longs.add(asset)
        elif current_price < low_10d:
            continue
        elif asset in context.long_positions:
            current_longs.add(asset)
    
    context.long_positions = current_longs
    
    if current_longs:
        weight = 1.0 / len(current_longs)
        pv = context.portfolio.portfolio_value
        
        for asset in context.assets:
            multiplier = context.multipliers[asset]
            
            current_price = data.current(asset, "close")
            if pd.isna(current_price):
                continue
                
            if asset in current_longs:
                target_value = weight * pv
                contracts = int(round(target_value / (current_price * multiplier)))
            else:
                contracts = 0
            
            current_position = context.portfolio.positions[asset].amount
            if contracts != current_position:
                order(asset, contracts - current_position)
    else:
        for asset in context.assets:
            current_position = context.portfolio.positions[asset].amount
            if current_position != 0:
                order(asset, -current_position)