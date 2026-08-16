"""Seed 8 / Island 8 (Benchmark): equal-weighted buy-and-hold.

Buy the equal-weighted portfolio once on the first bar and hold with no
rebalancing. This is the benchmark island's reference strategy.
"""

UNIVERSE = [
    "AAPL", "MSFT", "XOM", "GE", "CVX", "BRK", "PG", "PFE",
    "JNJ", "WFC", "JPM", "WMT", "BAC", "VZ", "ORCL",
]


def initialize(context):
    context.assets = [symbol(t) for t in UNIVERSE]
    context.entered = False


def handle_data(context, data):
    if context.entered:
        return
    context.entered = True
    weight = 1.0 / len(context.assets)
    for asset in context.assets:
        order_target_percent(asset, weight)
