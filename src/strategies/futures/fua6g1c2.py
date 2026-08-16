INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}

def initialize(context):
    context.assets = []
    context.multipliers = {}
    
    for sym in ["ES", "NQ"]:
        try:
            asset = symbol(sym)
            context.assets.append(asset)
            context.multipliers[asset] = INSTRUMENTS[sym]
        except:
            pass
    
    if not context.assets:
        return
    
    context.vol_window = 20
    schedule_function(rebalance, date_rules.week_start(), time_rules.market_open())

def rebalance(context, data):
    if not context.assets:
        return
    
    inv = {}
    
    for asset in context.assets:
        prices = safe_history(asset, 61, "1d")
        if len(prices) < 61:
            continue
            
        short_vol = prices[-21:-1].pct_change().std() * np.sqrt(BARS_PER_YEAR)
        long_vol = prices[-61:-1].pct_change().std() * np.sqrt(BARS_PER_YEAR)
        
        if short_vol > 0 and long_vol > 0:
            vol_of_vol = short_vol / long_vol
            dynamic_window = int(np.clip(20 * vol_of_vol, 5, 60))
            context.vol_window = dynamic_window
        
        lookback = context.vol_window
        window_prices = prices[-lookback-1:] if len(prices) >= lookback+1 else prices
        
        if len(window_prices) >= 2:
            vol = window_prices.pct_change().std() * np.sqrt(BARS_PER_YEAR)
            if vol > 0:
                inv[asset] = 1.0 / vol
    
    total = sum(inv.values())
    pv = context.portfolio.portfolio_value
    
    for asset in context.assets:
        if asset not in data:
            continue
        price = data.current(asset, "close")
        if price <= 0:
            continue
        w = inv.get(asset, 0.0) / total if total > 0 else 0.0
        contracts = int(round(w * pv / (price * context.multipliers[asset])))
        order(asset, contracts)