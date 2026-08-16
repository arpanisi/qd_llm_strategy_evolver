"""Baseline 4 (futures): MACD trend-following (12/26/9), rebalanced daily.

Identical formula to the equities version, applied to {ES, NQ}: go long
equal-weight among instruments whose MACD histogram is positive, otherwise
hold equal-weight cash-equivalent. Test-window-only baseline (Step 11).
"""

INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}  # symbol -> point value (USD/point)

WARMUP = 35  # 26 + 9 signal period


def initialize(context):
    context.instruments = {
        future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()
    }


def handle_data(context, data):
    longs = {}
    for asset, multiplier in context.instruments.items():
        prices = data.history(asset, "close", WARMUP, "1d")
        if len(prices) < WARMUP:
            continue
        ema12 = prices.ewm(span=12, adjust=False).mean()
        ema26 = prices.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        if macd.iloc[-1] > signal.iloc[-1]:
            longs[asset] = 1.0
    n = len(longs)
    pv = context.portfolio.portfolio_value
    for asset, multiplier in context.instruments.items():
        price = data.current(asset, "close")
        weight = longs.get(asset, 0.0) / n if n > 0 else 0.0
        contracts = int(round(weight * pv / (price * multiplier)))
        order_target_contracts(asset, contracts)
