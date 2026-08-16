"""Baseline 3 (equities): risk parity, rebalanced daily.

w_i = (1/sigma_i) / sum_j (1/sigma_j), where sigma_i is the rolling 60-day
annualized realized volatility of asset i (std of daily returns x sqrt of the
track's bars_per_year). Test-window-only baseline (Step 11).
"""

UNIVERSE = [
    "AAPL", "MSFT", "XOM", "GE", "CVX", "BRK", "PG", "PFE",
    "JNJ", "WFC", "JPM", "WMT", "BAC", "VZ", "ORCL",
]

LOOKBACK = 60
MIN_BARS = 30


def initialize(context):
    context.assets = [symbol(t) for t in UNIVERSE]


def handle_data(context, data):
    inv = {}
    total = 0.0
    for asset in context.assets:
        prices = data.history(asset, "close", LOOKBACK, "1d")
        if len(prices) < MIN_BARS:
            continue
        vol = prices.pct_change().std() * np.sqrt(BARS_PER_YEAR)
        if vol > 0:
            inv[asset] = 1.0 / vol
            total += 1.0 / vol
    for asset in context.assets:
        weight = inv.get(asset, 0.0) / total if total > 0 else 0.0
        order_target_percent(asset, weight)
