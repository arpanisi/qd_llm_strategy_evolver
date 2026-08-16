import numpy as np

INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}
ENTRY_WINDOW = 5
EXIT_WINDOW = 20

def initialize(context):
    context.instruments = {
        future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()
    }
    context.long_positions = set()
    context.warmup = max(ENTRY_WINDOW, EXIT_WINDOW) + 1

def handle_data(context, data):
    current_longs = set()
    
    for asset in context.instruments:
        prices = safe_history(asset, context.warmup, "1d")
        if len(prices) < context.warmup:
            continue
            
        current_close = prices.iloc[-1]
        prior_prices = prices.iloc[:-1]
        
        entry_high = prior_prices[-ENTRY_WINDOW:].max()
        exit_low = prior_prices[-EXIT_WINDOW:].min()
        
        if current_close > entry_high:
            current_longs.add(asset)
        elif current_close < exit_low:
            continue
        elif asset in context.long_positions:
            current_longs.add(asset)
    
    context.long_positions = current_longs
    
    portfolio_value = context.portfolio.portfolio_value
    active_count = len(current_longs)
    
    for asset, multiplier in context.instruments.items():
        current_price = data.current(asset, "close")
        if current_price is None or current_price <= 0:
            continue
            
        if asset in current_longs and active_count > 0:
            target_notional = portfolio_value / active_count
            target_contracts = int(round(target_notional / (current_price * multiplier)))
        else:
            target_contracts = 0
            
        current_position = context.portfolio.positions[asset].amount
        if target_contracts != current_position:
            order(asset, target_contracts - current_position)