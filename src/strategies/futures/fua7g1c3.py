import pandas as pd

INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}
START_DAY = 4
WINDOW_LENGTH = 10

def initialize(context):
    context.assets = [future_symbol(ticker) for ticker in INSTRUMENTS.keys()]
    context.multipliers = {asset: INSTRUMENTS[ticker] for asset, ticker in zip(context.assets, INSTRUMENTS.keys())}
    context.month = None
    context.day_in_month = 0
    context.active = False

def handle_data(context, data):
    dt = get_datetime()
    month = (dt.year, dt.month)
    
    if month != context.month:
        context.month = month
        context.day_in_month = 0
        context.active = False
    
    context.day_in_month += 1
    
    total_days = trading_days_per_month.get(month, 21)
    end_day = START_DAY + WINDOW_LENGTH - 1
    
    if START_DAY <= context.day_in_month <= min(end_day, total_days):
        context.active = True
    else:
        context.active = False
    
    pv = context.portfolio.portfolio_value
    for asset in context.assets:
        price = data.current(asset, "close")
        if price is None or price <= 0:
            continue
            
        multiplier = context.multipliers[asset]
        if context.active:
            target_contracts = int(round(pv / (len(context.assets) * price * multiplier)))
        else:
            target_contracts = 0
        
        current_position = context.portfolio.positions[asset].amount
        if target_contracts != current_position:
            order(asset, target_contracts - current_position)