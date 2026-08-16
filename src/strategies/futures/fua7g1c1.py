INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}

def initialize(context):
    context.instruments = {
        future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()
    }
    context.month = None
    context.day_in_month = 0

def handle_data(context, data):
    ts = data.current_dt
    month = (ts.year, ts.month)
    if month != context.month:
        context.month = month
        context.day_in_month = 0
    context.day_in_month += 1

    active = context.day_in_month == 1
    pv = context.portfolio.portfolio_value
    
    for asset, multiplier in context.instruments.items():
        price = data.current(asset, "close")
        if price is None:
            continue
        w = 1.0 / len(context.instruments) if active else 0.0
        contracts = int(round(w * pv / (price * multiplier)))
        current_pos = context.portfolio.positions[asset].amount
        if contracts != current_pos:
            order(asset, contracts - current_pos)