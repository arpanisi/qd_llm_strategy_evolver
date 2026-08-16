"""Seed 6 / Island 6 (Risk-Allocation / Risk Parity): inverse-20d-vol, weekly.

Weight each asset inversely proportional to its trailing 20-day annualized
realized volatility, renormalized to sum to 1. Same formula as Seed 2
(per coding-plan.md, the risk-allocation category is intentionally identical
to the volatility category at seed level).
"""

UNIVERSE = [
    "AAPL", "MSFT", "XOM", "GE", "CVX", "BRK", "PG", "PFE",
    "JNJ", "WFC", "JPM", "WMT", "BAC", "VZ", "ORCL",
]

VOL_LOOKBACK = 20


def initialize(context):
    context.assets = [symbol(t) for t in UNIVERSE]
    schedule_function(rebalance, date_rules.week_start(), time_rules.market_open())


def rebalance(context, data):
    inv = {}
    for asset in context.assets:
        prices = safe_history(asset, VOL_LOOKBACK + 1, "1d")
        if len(prices) >= VOL_LOOKBACK + 1:
            vol = prices.pct_change().std() * np.sqrt(BARS_PER_YEAR)
            if vol > 0:
                inv[asset] = 1.0 / vol
    total = sum(inv.values())
    for asset in context.assets:
        w = inv.get(asset, 0.0) / total if total > 0 else 0.0
        order_target_percent(asset, w)
