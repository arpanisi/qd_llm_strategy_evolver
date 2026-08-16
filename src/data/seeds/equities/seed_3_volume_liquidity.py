"""Seed 3 / Island 3 (Volume/Liquidity): volume spike tilt, daily.

Base equal weight across the universe; on any day an asset's volume exceeds
1.5x its trailing 20-day average volume it is given 2x the base weight.
Weights renormalized to sum to 1. Re-evaluated daily.
"""

UNIVERSE = [
    "AAPL", "MSFT", "XOM", "GE", "CVX", "BRK", "PG", "PFE",
    "JNJ", "WFC", "JPM", "WMT", "BAC", "VZ", "ORCL",
]

VOL_LOOKBACK = 20
VOL_SPIKE = 1.5
SPIKE_WEIGHT_MULT = 2.0


def initialize(context):
    context.assets = [symbol(t) for t in UNIVERSE]


def handle_data(context, data):
    base = 1.0 / len(context.assets)
    raw = {}
    for asset in context.assets:
        v = data.current(asset, "volume")
        hist = safe_history(asset, VOL_LOOKBACK + 1, "1d", field="volume")
        avg = hist.mean() if len(hist) >= VOL_LOOKBACK else None
        if avg and avg > 0 and v > VOL_SPIKE * avg:
            raw[asset] = SPIKE_WEIGHT_MULT * base
        else:
            raw[asset] = base
    total = sum(raw.values())
    for asset in context.assets:
        order_target_percent(asset, raw[asset] / total)
