"""Seed 5 / Island 5 (Statistical Arbitrage, futures variant): ES/NQ pair.

With only two instruments, the pairs leg is fixed: track the ES/NQ price
ratio against its 60-day mean. When it strays more than 2 sigma, go long the
underperformer and short the outperformer (dollar-neutral, each leg 50% of
notional); flat otherwise. Daily.
"""

INSTRUMENTS = {"ES": 50.0, "NQ": 20.0}  # symbol -> point value (USD/point)

RATIO_LOOKBACK = 61
Z = 2.0
LEG_WEIGHT = 0.5


def initialize(context):
    context.instruments = {
        future_symbol(sym): mult for sym, mult in INSTRUMENTS.items()
    }


def handle_data(context, data):
    es = next(a for a in context.instruments if a.symbol == "ES")
    nq = next(a for a in context.instruments if a.symbol == "NQ")

    pe = safe_history(es, RATIO_LOOKBACK, "1d")
    pn = safe_history(nq, RATIO_LOOKBACK, "1d")

    targets = {}
    if len(pe) >= RATIO_LOOKBACK and len(pn) >= RATIO_LOOKBACK:
        ratio_hist = pe / pn
        mu = ratio_hist.iloc[:-1].mean()
        sd = ratio_hist.iloc[:-1].std()
        ratio = ratio_hist.iloc[-1] if pn.iloc[-1] != 0 else 0.0
        if sd > 0:
            if ratio > mu + Z * sd:
                targets = {nq: LEG_WEIGHT, es: -LEG_WEIGHT}
            elif ratio < mu - Z * sd:
                targets = {es: LEG_WEIGHT, nq: -LEG_WEIGHT}

    pv = context.portfolio.portfolio_value
    for asset, multiplier in context.instruments.items():
        price = data.current(asset, "close")
        w = targets.get(asset, 0.0)
        contracts = int(round(w * pv / (price * multiplier)))
        order_target_contracts(asset, contracts)
