"""Baseline 5 (futures): RSI mean-reversion (14-day), rebalanced daily.

Identical formula to the equities version, applied to {ES, NQ}: among
oversold instruments (RSI < 30), weight w_i = (30 - RSI_i) / sum(30 - RSI_j);
if none are oversold, hold 100% cash-equivalent; RSI > 70 is never held long.
Test-window-only baseline (Step 11).
"""

INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}  # symbol -> point value (USD/point)

WARMUP = 15
PERIOD = 14


def initialize(context):
    context.instruments = {
        future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()
    }


def handle_data(context, data):
    w = {}
    for asset, multiplier in context.instruments.items():
        prices = data.history(asset, "close", WARMUP, "1d")
        if len(prices) < WARMUP:
            continue
        rsi = _rsi(prices, PERIOD)
        if rsi < 30:
            w[asset] = 30.0 - rsi
        elif rsi > 70:
            w[asset] = 0.0
    total = sum(w.values())
    pv = context.portfolio.portfolio_value
    for asset, multiplier in context.instruments.items():
        price = data.current(asset, "close")
        weight = w.get(asset, 0.0) / total if total > 0 else 0.0
        contracts = int(round(weight * pv / (price * multiplier)))
        order_target_contracts(asset, contracts)


def _rsi(close, period=PERIOD):
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs.iloc[-1]))
