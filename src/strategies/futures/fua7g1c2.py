def initialize(context):
    context.es = None
    context.nq = None
    context.multipliers = {}
    context.month = None
    context.day_in_month = 0
    
    for asset in context.universe:
        if asset.root_symbol == 'ES':
            context.es = asset
            context.multipliers[asset] = 50.0
        elif asset.root_symbol == 'NQ':
            context.nq = asset
            context.multipliers[asset] = 20.0

def handle_data(context, data):
    if context.es is None or context.nq is None:
        return
        
    ts = get_datetime()
    month = (ts.year, ts.month)
    
    if month != context.month:
        context.month = month
        context.day_in_month = 0
    context.day_in_month += 1
    
    total_days = trading_days_per_month.get(month, 21)
    start_active = 4
    end_active = total_days - 2
    
    active = (context.day_in_month >= start_active and 
              context.day_in_month <= end_active)
    
    pv = context.portfolio.portfolio_value
    
    for asset in [context.es, context.nq]:
        price = data.current(asset, "close")
        if price is None:
            continue
            
        multiplier = context.multipliers[asset]
        if active:
            target_contracts = int(round(0.5 * pv / (price * multiplier)))
        else:
            target_contracts = 0
            
        current_pos = context.portfolio.positions[asset].amount
        if target_contracts != current_pos:
            order(asset, target_contracts - current_pos)