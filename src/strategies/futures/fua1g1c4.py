INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}
RSI_PERIOD = 14
RSI_OVERSOLD = 30.0
MA_SHORT = 50
MA_LONG = 200
WARMUP = max(RSI_PERIOD, MA_LONG) + 1

def initialize(context):
    context.instruments = {future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()}

def handle_data(context, data):
    longs = []
    for asset in context.instruments:
        prices = safe_history(asset, WARMUP, "1d")
        if len(prices) < WARMUP:
            continue
        rsi_val = _rsi(prices, RSI_PERIOD).iloc[-1]
        ma_short = prices[-MA_SHORT:].mean()
        ma_long = prices[-MA_LONG:].mean()
        if rsi_val < RSI_OVERSOLD and ma_short > ma_long:
            longs.append(asset)
    
    n = len(longs)
    pv = context.portfolio.portfolio_value
    for asset, multiplier in context.instruments.items():
        price = data.current(asset, "close")
        if price is None:
            continue
        weight = 1.0 / n if asset in longs else 0.0
        contracts = int(round(weight * pv / (price * multiplier)))
        order(asset, contracts)

def _rsi(prices, period):
    delta = prices.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1.0/period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)