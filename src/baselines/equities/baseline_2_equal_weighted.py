"""Baseline 2 (equities): equal-weighted, rebalanced daily.

This is the Information Ratio benchmark R_b (Step 8A/11). Rebalanced daily,
so it is computable over train, validation, and test windows with no fixed
start point.
"""

UNIVERSE = [
    "AAPL", "MSFT", "XOM", "GE", "CVX", "BRK", "PG", "PFE",
    "JNJ", "WFC", "JPM", "WMT", "BAC", "VZ", "ORCL",
]


def initialize(context):
    context.assets = [symbol(t) for t in UNIVERSE]
    context.weight = 1.0 / len(context.assets)


def handle_data(context, data):
    for asset in context.assets:
        order_target_percent(asset, context.weight)
