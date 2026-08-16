"""Seed 3 / Island 3 (Volume/Liquidity, futures variant): volume spike, daily.

Base equal weight across {ES, NQ}; on any day an instrument's volume exceeds
1.5x its trailing 20-day average volume it gets 2x the base weight. Weights
renormalized to sum to 1, converted to whole contracts. Daily.
"""

INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}  # symbol -> point value (USD/point)

VOL_LOOKBACK = 20
VOL_SPIKE = 1.5
SPIKE_WEIGHT_MULT = 2.0


def initialize(context):
    context.instruments = {
        future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()
    }


def handle_data(context, data):
    base = 1.0 / len(context.instruments)
    raw = {}
    for asset in context.instruments:
        v = data.current(asset, "volume")
        hist = safe_history(asset, VOL_LOOKBACK + 1, "1d", field="volume")
        avg = hist.mean() if len(hist) >= VOL_LOOKBACK else None
        if avg and avg > 0 and v > VOL_SPIKE * avg:
            raw[asset] = SPIKE_WEIGHT_MULT * base
        else:
            raw[asset] = base
    total = sum(raw.values())
    pv = context.portfolio.portfolio_value
    for asset, multiplier in context.instruments.items():
        price = data.current(asset, "close")
        contracts = int(round((raw[asset] / total) * pv / (price * multiplier)))
        order_target_contracts(asset, contracts)
