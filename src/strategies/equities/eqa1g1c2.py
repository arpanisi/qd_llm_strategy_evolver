UNIVERSE = ["AAPL", "MSFT", "XOM", "GE", "CVX", "BRK", "PG", "PFE", "JNJ", "WFC", "JPM", "WMT", "BAC", "VZ", "ORCL"]
RSI_PERIOD = 14
RSI_BUY = 25.0
RSI_SELL = 35.0
MIN_HOLD = 3
WARMUP = RSI_PERIOD + 1

def initialize(context):
    context.assets = [symbol(t) for t in UNIVERSE]
    context.entry_dates = {}

def handle_data(context, data):
    longs = []
    now = context.get_datetime()
    today = now.date()
    
    for asset in context.assets:
        prices = safe_history(asset, WARMUP, "1d")
        if len(prices) < WARMUP:
            continue
            
        rsi = _rsi(prices, RSI_PERIOD).iloc[-1]
        
        if asset in context.entry_dates:
            held = (today - context.entry_dates[asset]).days
            if held >= MIN_HOLD and rsi > RSI_SELL:
                del context.entry_dates[asset]
                continue
                
        if rsi < RSI_BUY:
            longs.append(asset)
            context.entry_dates[asset] = today
            
    weight = 1.0 / len(longs) if longs else 0.0
    for asset in context.assets:
        order_target_percent(asset, weight if asset in longs else 0.0)

def _rsi(prices, period):
    delta = prices.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)