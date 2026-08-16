INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}
RSI_PERIOD = 14
RSI_OVERSOLD = 30.0
WARMUP = RSI_PERIOD + 1

def initialize(context):
    context.instruments = {future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()}

def handle_data(context, data):
    rsi_values = {}
    valid_data = True
    
    for asset in context.instruments:
        prices = safe_history(asset, WARMUP, "1d")
        if len(prices) < WARMUP:
            valid_data = False
            break
        rsi = _rsi(prices, RSI_PERIOD).iloc[-1]
        rsi_values[asset] = rsi
    
    if not valid_data:
        return
    
    oversold_assets = [asset for asset, rsi in rsi_values.items() if rsi < RSI_OVERSOLD]
    
    if len(oversold_assets) < 2:
        for asset in context.instruments:
            order(asset, 0)
        return
    
    distances = {asset: RSI_OVERSOLD - rsi_values[asset] for asset in oversold_assets}
    total_distance = sum(distances.values())
    
    if total_distance <= 0:
        for asset in context.instruments:
            order(asset, 0)
        return
    
    pv = context.portfolio.portfolio_value
    target_contracts = {}
    
    for asset in context.instruments:
        if asset in oversold_assets:
            weight = distances[asset] / total_distance
            price = data.current(asset, "close")
            multiplier = context.instruments[asset]
            contracts = int(round(weight * pv / (price * multiplier)))
            target_contracts[asset] = max(contracts, 1)
        else:
            target_contracts[asset] = 0
    
    for asset, target in target_contracts.items():
        current = context.portfolio.positions[asset].amount if asset in context.portfolio.positions else 0
        if target != current:
            order(asset, target - current)

def _rsi(prices, period):
    delta = prices.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1.0/period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)