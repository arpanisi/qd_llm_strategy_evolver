INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}
WARMUP = -1
LOOKBACK = 20
CONFIRMATION_BUFFER = 0.02
COOLDOWN_DAYS = 5

def initialize(context):
    context.instruments = {future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()}
    context.long = set()
    context.exit_dates = {}
    context.warmup_done = False
    context.warmup_counter = 0

def handle_data(context, data):
    context.warmup_counter += 1
    if context.warmup_counter < LOOKBACK + 1:
        return
    if not context.warmup_done:
        context.warmup_done = True

    current_date = get_datetime()
    longs = set()
    
    for asset in context.instruments:
        prices = safe_history(asset, LOOKBACK + 1, "1d")
        if len(prices) < LOOKBACK + 1:
            continue
            
        prior = prices.iloc[:-1]
        close = prices.iloc[-1]
        hi = prior.max()
        lo = prior.min()
        
        in_cooldown = False
        if asset in context.exit_dates:
            days_since_exit = (current_date - context.exit_dates[asset]).days
            if days_since_exit < COOLDOWN_DAYS:
                in_cooldown = True
        
        entry_signal = close > hi * (1 + CONFIRMATION_BUFFER)
        exit_signal = close < lo
        
        if entry_signal and not in_cooldown:
            longs.add(asset)
        elif exit_signal:
            if asset in context.long:
                context.exit_dates[asset] = current_date
            continue
        elif asset in context.long and not in_cooldown:
            longs.add(asset)
    
    context.long = longs
    n = len(longs)
    pv = context.portfolio.portfolio_value
    
    for asset, multiplier in context.instruments.items():
        price = data.current(asset, "close")
        if price is None:
            continue
        weight = 1.0 / n if asset in longs else 0.0
        contracts = int(round(weight * pv / (price * multiplier)))
        order(asset, contracts)