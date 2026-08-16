INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}
RSI_PERIOD = 14
RSI_ENTRY = 25.0
RSI_EXIT = 35.0
WARMUP = RSI_PERIOD + 1

def initialize(context):
    context.assets = [future_symbol("ES"), future_symbol("NQ")]
    context.multipliers = {future_symbol("ES"): 50.0, future_symbol("NQ"): 20.0}
    context.positions = {}

def handle_data(context, data):
    for asset in context.assets:
        prices = safe_history(asset, WARMUP, "1d")
        if len(prices) < WARMUP:
            continue
            
        rsi_series = _rsi(prices, RSI_PERIOD)
        current_rsi = rsi_series.iloc[-1]
        prev_rsi = rsi_series.iloc[-2] if len(rsi_series) > 1 else 100.0
        
        current_pos = context.positions.get(asset, 0)
        
        if current_pos == 0 and current_rsi < RSI_ENTRY:
            pv = context.portfolio.portfolio_value
            price = data.current(asset, "close")
            multiplier = context.multipliers[asset]
            contracts = int(round(pv / (2.0 * price * multiplier)))
            if contracts > 0:
                order(asset, contracts)
                context.positions[asset] = contracts
                
        elif current_pos > 0 and prev_rsi < RSI_EXIT and current_rsi >= RSI_EXIT:
            order(asset, -current_pos)
            context.positions[asset] = 0

def _rsi(prices, period):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))