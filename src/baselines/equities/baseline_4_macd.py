"""Baseline 4 (equities): MACD trend-following (12/26/9), rebalanced daily.

Go long equal-weight among assets whose MACD histogram (MACD line minus its
9-day signal line) is positive; hold equal-weight cash-equivalent otherwise.
Test-window-only baseline (Step 11).
"""

UNIVERSE = [
    "AAPL", "MSFT", "XOM", "GE", "CVX", "BRK", "PG", "PFE",
    "JNJ", "WFC", "JPM", "WMT", "BAC", "VZ", "ORCL",
]

WARMUP = 35  # 26 + 9 signal period


def initialize(context):
    context.assets = [symbol(t) for t in UNIVERSE]


def handle_data(context, data):
    longs = {}
    n = 0
    for asset in context.assets:
        prices = data.history(asset, "close", WARMUP, "1d")
        if len(prices) < WARMUP:
            continue
        ema12 = prices.ewm(span=12, adjust=False).mean()
        ema26 = prices.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        if macd.iloc[-1] > signal.iloc[-1]:
            longs[asset] = 1.0
            n += 1
    for asset in context.assets:
        weight = longs.get(asset, 0.0) / n if n > 0 else 0.0
        order_target_percent(asset, weight)
