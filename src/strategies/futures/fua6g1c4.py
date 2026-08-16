INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}
VOL_LOOKBACK = 20
MOM_LOOKBACK = 5

def initialize(context):
    context.instruments = {future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()}
    schedule_function(rebalance, date_rules.week_start(), time_rules.market_open())

def rebalance(context, data):
    inv = {}
    mom_ok = {}
    for asset in context.instruments:
        prices = safe_history(asset, max(VOL_LOOKBACK, MOM_LOOKBACK) + 1, "1d")
        if len(prices) >= VOL_LOOKBACK + 1:
            vol = prices.pct_change().std() * np.sqrt(BARS_PER_YEAR)
            if vol > 0:
                inv[asset] = 1.0 / vol
        if len(prices) >= MOM_LOOKBACK + 1:
            mom_ok[asset] = prices.iloc[-1] > prices.iloc[-MOM_LOOKBACK-1]
    
    total = sum(inv.values())
    pv = context.portfolio.portfolio_value
    trade = all(mom_ok.values()) and len(mom_ok) == len(context.instruments)
    
    for asset, multiplier in context.instruments.items():
        price = data.current(asset, "close")
        if trade and total > 0:
            w = inv.get(asset, 0.0) / total
        else:
            w = 0.0
        contracts = int(round(w * pv / (price * multiplier)))
        order(asset, contracts - context.portfolio.positions[asset].amount)