INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}
RSI_PERIOD = 14
RSI_OVERSOLD = 30.0
MA_PERIOD = 50
WARMUP = max(RSI_PERIOD, MA_PERIOD) + 1

def initialize(context):
    context.assets = []
    context.multipliers = {}
    for sym in ["ES", "NQ"]:
        try:
            asset = symbol(sym)
            context.assets.append(asset)
            context.multipliers[asset] = INSTRUMENTS[sym]
        except:
            pass

def handle_data(context, data):
    longs = []
    for asset in context.assets:
        prices = safe_history(asset, WARMUP, "1d")
        if len(prices) < WARMUP:
            continue
        rsi_val = _rsi(prices, RSI_PERIOD).iloc[-1]
        ma = prices[-MA_PERIOD:].mean()
        current_price = data.current(asset, "close")
        if rsi_val < RSI_OVERSOLD and current_price > ma:
            longs.append(asset)
    
    n = len(longs)
    pv = context.portfolio.portfolio_value
    for asset in context.assets:
        price = data.current(asset, "close")
        multiplier = context.multipliers[asset]
        weight = 1.0 / n if asset in longs else 0.0
        contracts = int(round(weight * pv / (price * multiplier))) if n > 0 else 0
        current_pos = context.portfolio.positions[asset].amount
        if contracts != current_pos:
            order(asset, contracts - current_pos)

def _rsi(prices, period):
    delta = prices.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1.0/period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)