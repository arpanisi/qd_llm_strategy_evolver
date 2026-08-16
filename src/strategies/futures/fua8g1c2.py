def initialize(context):
    context.es = future_symbol("ES")
    context.nq = future_symbol("NQ")
    context.multipliers = {context.es: 50.0, context.nq: 20.0}
    
    schedule_function(rebalance, date_rules.month_start(), time_rules.market_open())

def rebalance(context, data):
    es_price = data.current(context.es, "close")
    nq_price = data.current(context.nq, "close")
    
    if es_price is None or nq_price is None:
        return
    
    es_notional = context.portfolio.positions[context.es].amount * es_price * context.multipliers[context.es]
    nq_notional = context.portfolio.positions[context.nq].amount * nq_price * context.multipliers[context.nq]
    
    target_notional = context.portfolio.portfolio_value / 2.0
    
    es_diff = target_notional - es_notional
    nq_diff = target_notional - nq_notional
    
    if abs(es_diff) > 0:
        es_contracts = int(round(es_diff / (es_price * context.multipliers[context.es])))
        if es_contracts != 0:
            order(context.es, es_contracts)
    
    if abs(nq_diff) > 0:
        nq_contracts = int(round(nq_diff / (nq_price * context.multipliers[context.nq])))
        if nq_contracts != 0:
            order(context.nq, nq_contracts)

def handle_data(context, data):
    pass