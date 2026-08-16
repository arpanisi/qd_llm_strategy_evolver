import numpy as np

INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}
VOL_LOOKBACK = 20
EWMA_SPAN = 20
REBALANCE_FREQ = 'week_start'

def initialize(context):
    context.instruments = {
        future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()
    }
    context.vol_cache = {}
    schedule_function(
        rebalance,
        date_rules.week_start(),
        time_rules.market_open()
    )

def ewma_volatility(returns_series, span):
    if len(returns_series) < 2:
        return 0.0
    variance = returns_series.ewm(span=span, adjust=False).var()
    if variance.iloc[-1] <= 0:
        return 0.0
    return np.sqrt(variance.iloc[-1] * BARS_PER_YEAR)

def rebalance(context, data):
    inv_vol = {}
    for asset in context.instruments:
        prices = safe_history(asset, VOL_LOOKBACK + 1, "1d")
        if len(prices) < VOL_LOOKBACK + 1:
            continue
        returns = prices.pct_change().dropna()
        vol = ewma_volatility(returns, EWMA_SPAN)
        if vol > 0:
            inv_vol[asset] = 1.0 / vol
    
    total = sum(inv_vol.values())
    if total <= 0:
        return
    
    pv = context.portfolio.portfolio_value
    for asset, multiplier in context.instruments.items():
        price = data.current(asset, "close")
        if price is None or price <= 0:
            continue
        weight = inv_vol.get(asset, 0.0) / total
        target_value = weight * pv
        contracts = int(round(target_value / (price * multiplier)))
        current = context.portfolio.positions[asset].amount
        if contracts != current:
            order(asset, contracts - current)