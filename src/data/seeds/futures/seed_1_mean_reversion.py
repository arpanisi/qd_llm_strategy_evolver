"""Seed 1 / Island 1 (Mean-Reversion, futures variant): RSI-14 oversold, daily.

Each day, hold equal notional weight in whichever of ES/NQ has a 14-day RSI
below 30 (oversold); hold cash if neither is oversold. Re-evaluated daily.
"""

INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}  # symbol -> point value (USD/point)

RSI_PERIOD = 14
RSI_OVERSOLD = 30.0
WARMUP = RSI_PERIOD + 1


def initialize(context):
    context.instruments = {
        future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()
    }


def handle_data(context, data):
    longs = []
    for asset in context.instruments:
        prices = safe_history(asset, WARMUP, "1d")
        if len(prices) >= WARMUP and _rsi(prices, RSI_PERIOD).iloc[-1] < RSI_OVERSOLD:
            longs.append(asset)
    n = len(longs)
    pv = context.portfolio.portfolio_value
    for asset, multiplier in context.instruments.items():
        price = data.current(asset, "close")
        weight = 1.0 / n if asset in longs else 0.0
        contracts = int(round(weight * pv / (price * multiplier)))
        order_target_contracts(asset, contracts)


def _rsi(prices, period):
    delta = prices.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)
