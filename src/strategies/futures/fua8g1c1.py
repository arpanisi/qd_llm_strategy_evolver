import numpy as np

INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}
VOL_WINDOW = 20
TARGET_RISK_CONTRIBUTION = 0.5

def initialize(context):
    context.assets = {future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()}
    context.vol_window = VOL_WINDOW
    context.target_risk = TARGET_RISK_CONTRIBUTION
    
    schedule_function(rebalance, date_rules.every_day(), time_rules.market_open())

def compute_volatility(asset, data, window):
    prices = safe_history(asset, window + 1, "1d")
    if len(prices) < 2:
        return None
    returns = np.log(prices / prices.shift(1)).dropna()
    if len(returns) < window:
        return None
    return returns.std() * np.sqrt(BARS_PER_YEAR)

def rebalance(context, data):
    volatilities = {}
    valid_assets = []
    
    for asset in context.assets:
        vol = compute_volatility(asset, data, context.vol_window)
        if vol is not None and vol > 0:
            volatilities[asset] = vol
            valid_assets.append(asset)
    
    if len(valid_assets) != 2:
        return
    
    es_asset = future_symbol("ES")
    nq_asset = future_symbol("NQ")
    
    es_vol = volatilities.get(es_asset)
    nq_vol = volatilities.get(nq_asset)
    
    if es_vol is None or nq_vol is None:
        return
    
    total_vol_inv = 1/es_vol + 1/nq_vol
    es_weight = (1/es_vol) / total_vol_inv
    nq_weight = (1/nq_vol) / total_vol_inv
    
    portfolio_value = context.portfolio.portfolio_value
    
    for asset, target_weight in [(es_asset, es_weight), (nq_asset, nq_weight)]:
        current_price = data.current(asset, "close")
        multiplier = context.assets[asset]
        
        if current_price <= 0 or multiplier <= 0:
            continue
            
        target_notional = portfolio_value * target_weight
        target_contracts = int(round(target_notional / (current_price * multiplier)))
        
        current_position = context.portfolio.positions[asset].amount
        if target_contracts != current_position:
            order(asset, target_contracts - current_position)
    
    record(es_weight=es_weight, nq_weight=nq_weight, es_vol=es_vol, nq_vol=nq_vol)