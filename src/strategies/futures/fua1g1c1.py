INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}
RSI_PERIOD = 14
ENTRY_THRESHOLD = 25.0
EXIT_THRESHOLD = 45.0
WARMUP = RSI_PERIOD + 1

def initialize(context):
    context.instruments = {future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()}
    context.positions = {asset: 0 for asset in context.instruments}

def handle_data(context, data):
    for asset, multiplier in context.instruments.items():
        prices = safe_history(asset, WARMUP, "1d")
        if len(prices) < WARMUP:
            continue
            
        current_rsi = _rsi(prices, RSI_PERIOD).iloc[-1]
        current_price = data.current(asset, "close")
        
        if current_price is None:
            continue
            
        current_contracts = context.positions.get(asset, 0)
        
        if current_contracts == 0:
            if current_rsi < ENTRY_THRESHOLD:
                pv = context.portfolio.portfolio_value
                weight = 1.0 / len(context.instruments)
                target_contracts = int(round(weight * pv / (current_price * multiplier)))
                if target_contracts != 0:
                    order(asset, target_contracts)
                    context.positions[asset] = target_contracts
        else:
            if current_rsi > EXIT_THRESHOLD:
                order(asset, -current_contracts)
                context.positions[asset] = 0

def _rsi(prices, period):
    delta = prices.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1.0/period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)