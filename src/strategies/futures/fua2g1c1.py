INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}
VOL_LOOKBACK = 20
MOM_LOOKBACK = 10

def initialize(context):
    context.instruments = {
        future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()
    }
    schedule_function(rebalance, date_rules.week_start(), time_rules.market_open())

def rebalance(context, data):
    inv_vol = {}
    momentum = {}
    
    for asset in context.instruments:
        prices = safe_history(asset, max(VOL_LOOKBACK, MOM_LOOKBACK) + 1, "1d")
        
        if len(prices) >= VOL_LOOKBACK + 1:
            vol = prices.pct_change().std() * np.sqrt(BARS_PER_YEAR)
            if vol > 0:
                inv_vol[asset] = 1.0 / vol
        
        if len(prices) >= MOM_LOOKBACK + 1:
            momentum[asset] = (prices.iloc[-1] / prices.iloc[-MOM_LOOKBACK] - 1.0)
    
    if not inv_vol or not momentum:
        for asset in context.instruments:
            order(asset, 0)
        return
    
    any_positive_momentum = any(m > 0 for m in momentum.values())
    
    if not any_positive_momentum:
        for asset in context.instruments:
            order(asset, 0)
        return
    
    total_inv_vol = sum(inv_vol.values())
    pv = context.portfolio.portfolio_value
    
    for asset, multiplier in context.instruments.items():
        price = data.current(asset, "close")
        if price and asset in inv_vol:
            w = inv_vol[asset] / total_inv_vol
            contracts = int(round(w * pv / (price * multiplier)))
            order(asset, contracts)
        else:
            order(asset, 0)