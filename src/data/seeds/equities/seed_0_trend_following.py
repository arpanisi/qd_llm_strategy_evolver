"""Seed 0 / Island 0 (Trend-Following): momentum top-3, monthly.

Each month, buy the 3 assets with the highest trailing 90-day return,
equal-weighted (1/3 each), and rebalance monthly. Cash otherwise.
"""

UNIVERSE = [
    "AAPL", "MSFT", "XOM", "GE", "CVX", "BRK", "PG", "PFE",
    "JNJ", "WFC", "JPM", "WMT", "BAC", "VZ", "ORCL",
]

TOP_N = 3
WARMUP = 91  # 90 returns + 1 starting close


def initialize(context):
    context.assets = [symbol(t) for t in UNIVERSE]
    schedule_function(rebalance, date_rules.month_start(), time_rules.market_open())


def rebalance(context, data):
    rets = {}
    for asset in context.assets:
        prices = safe_history(asset, WARMUP, "1d")
        if len(prices) >= WARMUP:
            rets[asset] = prices.iloc[-1] / prices.iloc[0] - 1.0
    ranked = sorted(rets, key=rets.get, reverse=True)[:TOP_N]
    weight = 1.0 / TOP_N if len(ranked) == TOP_N else 0.0
    for asset in context.assets:
        order_target_percent(asset, weight if asset in ranked else 0.0)
