"""Baseline 5 (equities): RSI mean-reversion (14-day), rebalanced daily.

Among assets currently oversold (RSI < 30), weight w_i = (30 - RSI_i) /
sum_j (30 - RSI_j), so more-oversold assets get more weight. If no asset is
oversold, hold 100% cash-equivalent. Assets with RSI > 70 are never held long.
Test-window-only baseline (Step 11).
"""

UNIVERSE = [
    "AAPL", "MSFT", "XOM", "GE", "CVX", "BRK", "PG", "PFE",
    "JNJ", "WFC", "JPM", "WMT", "BAC", "VZ", "ORCL",
]

WARMUP = 15
PERIOD = 14


def initialize(context):
    context.assets = [symbol(t) for t in UNIVERSE]


def handle_data(context, data):
    w = {}
    for asset in context.assets:
        prices = data.history(asset, "close", WARMUP, "1d")
        if len(prices) < WARMUP:
            continue
        rsi = _rsi(prices, PERIOD)
        if rsi < 30:
            w[asset] = 30.0 - rsi
        elif rsi > 70:
            w[asset] = 0.0
    total = sum(w.values())
    for asset in context.assets:
        weight = w.get(asset, 0.0) / total if total > 0 else 0.0
        order_target_percent(asset, weight)


def _rsi(close, period=PERIOD):
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs.iloc[-1]))
