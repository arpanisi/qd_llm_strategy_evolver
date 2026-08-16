"""Seed 4 / Island 4 (Breakout, futures variant): 20-day high/low breakout.

Per instrument: go long when close exceeds the trailing 20-day high; exit
when close falls below the trailing 20-day low. Equal notional weight among
currently-long instruments, converted to whole contracts. Daily.
"""

INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}  # symbol -> point value (USD/point)

WARMUP = 21  # 20 prior bars + current


def initialize(context):
    context.instruments = {
        future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()
    }
    context.long = set()


def handle_data(context, data):
    longs = set()
    for asset in context.instruments:
        prices = safe_history(asset, WARMUP, "1d")
        if len(prices) < WARMUP:
            continue
        prior = prices.iloc[:-1]
        close = prices.iloc[-1]
        hi = prior.max()
        lo = prior.min()
        if close > hi:
            longs.add(asset)
        elif close < lo:
            continue
        elif asset in context.long:
            longs.add(asset)
    context.long = longs
    n = len(longs)
    pv = context.portfolio.portfolio_value
    for asset, multiplier in context.instruments.items():
        price = data.current(asset, "close")
        weight = 1.0 / n if asset in longs else 0.0
        contracts = int(round(weight * pv / (price * multiplier)))
        order_target_contracts(asset, contracts)
