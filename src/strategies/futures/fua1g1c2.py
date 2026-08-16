def initialize(context):
    context.es = future_symbol("ES")
    context.nq = future_symbol("NQ")
    context.instruments = {
        context.es: 50.0,
        context.nq: 20.0
    }
    context.rsi_period = 14
    context.rsi_oversold = 30.0
    context.ma_period = 50
    context.warmup = max(context.rsi_period, context.ma_period) + 1

def handle_data(context, data):
    longs = []
    for asset, multiplier in context.instruments.items():
        prices = safe_history(asset, context.warmup, "1d")
        if len(prices) < context.warmup:
            continue
            
        current_price = data.current(asset, "close")
        if current_price is None:
            continue
            
        rsi_series = _rsi(prices, context.rsi_period)
        if len(rsi_series) == 0:
            continue
            
        rsi = rsi_series.iloc[-1]
        ma = prices.rolling(context.ma_period).mean().iloc[-1]
        
        if pd.isna(rsi) or pd.isna(ma):
            continue
            
        if rsi < context.rsi_oversold and current_price > ma:
            longs.append(asset)
    
    n = len(longs)
    pv = context.portfolio.portfolio_value
    
    for asset, multiplier in context.instruments.items():
        price = data.current(asset, "close")
        if price is None:
            continue
            
        if n > 0 and asset in longs:
            target_contracts = int(round(pv / (n * price * multiplier)))
        else:
            target_contracts = 0
        
        current_position = context.portfolio.positions[asset].amount
        if current_position != target_contracts:
            order(asset, target_contracts - current_position)

def _rsi(prices, period):
    delta = prices.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1.0/period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)