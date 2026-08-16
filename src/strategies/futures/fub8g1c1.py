import numpy as np

INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}
LOOKBACK = 5
THRESHOLD = 0.02

def initialize(context):
    context.assets = {future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()}
    context.last_signal = 0
    schedule_function(rebalance, date_rules.every_day(), time_rules.market_open())

def rebalance(context, data):
    assets = list(context.assets.keys())
    
    prices = {}
    for asset in assets:
        price = data.current(asset, "close")
        if price is None or np.isnan(price) or price <= 0:
            return
        prices[asset] = price
    
    returns = {}
    for asset in assets:
        hist = safe_history(asset, LOOKBACK + 1, "1d")
        if len(hist) < LOOKBACK + 1:
            return
        ret = hist.pct_change().dropna()
        if len(ret) < LOOKBACK:
            return
        returns[asset] = ret
    
    es_ret = returns[assets[0]].iloc[-LOOKBACK:].mean()
    nq_ret = returns[assets[1]].iloc[-LOOKBACK:].mean()
    spread = es_ret - nq_ret
    
    signal = 0
    if spread > THRESHOLD:
        signal = -1
    elif spread < -THRESHOLD:
        signal = 1
    
    if signal == context.last_signal:
        return
    
    context.last_signal = signal
    
    for asset in assets:
        order_target(asset, 0)
    
    if signal == 0:
        return
    
    if signal == -1:
        long_asset = assets[1]
        short_asset = assets[0]
    else:
        long_asset = assets[0]
        short_asset = assets[1]
    
    long_price = prices[long_asset]
    short_price = prices[short_asset]
    long_mult = context.assets[long_asset]
    short_mult = context.assets[short_asset]
    
    notional = context.portfolio.portfolio_value * 0.5
    long_contracts = int(round(notional / (long_price * long_mult)))
    short_contracts = int(round(notional / (short_price * short_mult)))
    
    if long_contracts != 0:
        order(long_asset, long_contracts)
    if short_contracts != 0:
        order(short_asset, -short_contracts)