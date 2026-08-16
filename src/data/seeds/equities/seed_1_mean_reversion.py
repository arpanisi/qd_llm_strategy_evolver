"""Seed 1 / Island 1 (Mean-Reversion): RSI-14 oversold, daily.

Each day, hold equal weight in every asset whose 14-day RSI is below 30
(oversold); everything else in cash. Re-evaluated daily.
"""

UNIVERSE = [
    "AAPL", "MSFT", "XOM", "GE", "CVX", "BRK", "PG", "PFE",
    "JNJ", "WFC", "JPM", "WMT", "BAC", "VZ", "ORCL",
]

RSI_PERIOD = 14
RSI_OVERSOLD = 30.0
WARMUP = RSI_PERIOD + 1


def initialize(context):
    context.assets = [symbol(t) for t in UNIVERSE]


def handle_data(context, data):
    longs = []
    for asset in context.assets:
        prices = safe_history(asset, WARMUP, "1d")
        if len(prices) >= WARMUP and _rsi(prices, RSI_PERIOD).iloc[-1] < RSI_OVERSOLD:
            longs.append(asset)
    weight = 1.0 / len(longs) if longs else 0.0
    for asset in context.assets:
        order_target_percent(asset, weight if asset in longs else 0.0)


def _rsi(prices, period):
    delta = prices.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)
