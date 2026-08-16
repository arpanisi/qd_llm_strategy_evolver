"""Seed 4 / Island 4 (Breakout/Pattern): 20-day high/low breakout, daily.

Per asset: go long when close exceeds the trailing 20-day high; exit when
close falls below the trailing 20-day low. Equal weight among currently-long
assets. State (currently-long set) is carried in context.
"""

UNIVERSE = [
    "AAPL", "MSFT", "XOM", "GE", "CVX", "BRK", "PG", "PFE",
    "JNJ", "WFC", "JPM", "WMT", "BAC", "VZ", "ORCL",
]

WARMUP = 21  # 20 prior bars + current


def initialize(context):
    context.assets = [symbol(t) for t in UNIVERSE]
    context.long = set()


def handle_data(context, data):
    longs = set()
    for asset in context.assets:
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
    weight = 1.0 / len(longs) if longs else 0.0
    for asset in context.assets:
        order_target_percent(asset, weight if asset in longs else 0.0)
