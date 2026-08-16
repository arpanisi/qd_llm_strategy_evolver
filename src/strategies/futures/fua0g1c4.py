INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}
WARMUP = 181

def initialize(context):
    context.instruments = {future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()}
    schedule_function(rebalance, date_rules.month_start(), time_rules.market_open())

def rebalance(context, data):
    rets = {}
    ma_filter = {}
    
    for asset in context.instruments:
        prices = safe_history(asset, WARMUP, "1d")
        if len(prices) >= WARMUP:
            ret_90 = prices.iloc[-1] / prices.iloc[-91] - 1.0
            ma_180 = (prices.iloc[-1] / prices.iloc[-181] - 1.0) * (90/180)
            rets[asset] = ret_90
            ma_filter[asset] = ret_90 > ma_180
    
    pv = context.portfolio.portfolio_value
    valid_assets = [a for a in rets if rets[a] > 0 and ma_filter.get(a, False)]
    
    for asset, multiplier in context.instruments.items():
        price = data.current(asset, "close")
        if valid_assets and rets.get(asset, -1) == max(rets[a] for a in valid_assets):
            contracts = int(round(pv / (price * multiplier)))
        else:
            contracts = 0
        order(asset, contracts)