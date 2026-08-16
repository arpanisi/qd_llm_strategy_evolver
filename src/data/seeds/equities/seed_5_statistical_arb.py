"""Seed 5 / Island 5 (Statistical Arbitrage / Pairs): 2-sigma ratio band.

Each day, pick the two assets with the highest trailing 90-day return
correlation. When their price ratio strays more than 2 sigma from its 60-day
mean, go long the underperformer and short the outperformer (dollar-neutral,
+/-50% of portfolio value each); flat otherwise.
"""

UNIVERSE = [
    "AAPL", "MSFT", "XOM", "GE", "CVX", "BRK", "PG", "PFE",
    "JNJ", "WFC", "JPM", "WMT", "BAC", "VZ", "ORCL",
]

CORR_LOOKBACK = 91
RATIO_LOOKBACK = 61
Z = 2.0
LEG_WEIGHT = 0.5


def initialize(context):
    context.assets = [symbol(t) for t in UNIVERSE]


def handle_data(context, data):
    closes = {}
    for asset in context.assets:
        closes[asset] = safe_history(asset, CORR_LOOKBACK, "1d")

    rets = pd.DataFrame(
        {a: closes[a].pct_change() for a in context.assets}
    ).dropna()
    targets = {}
    if len(rets) >= 30:
        corr = rets.corr()
        best = None
        best_c = -2.0
        n = len(context.assets)
        for i in range(n):
            for j in range(i + 1, n):
                a = context.assets[i]
                b = context.assets[j]
                c = corr.loc[a, b]
                if c == c and c > best_c:
                    best_c, best = c, (a, b)
        if best is not None:
            a, b = best
            pa = closes[a]
            pb = closes[b]
            ratio_hist = (pa / pb).iloc[-RATIO_LOOKBACK:]
            mu = ratio_hist.mean()
            sd = ratio_hist.std()
            ratio = (pa.iloc[-1] / pb.iloc[-1]) if pb.iloc[-1] != 0 else 0.0
            if sd > 0:
                if ratio > mu + Z * sd:
                    targets = {b: LEG_WEIGHT, a: -LEG_WEIGHT}
                elif ratio < mu - Z * sd:
                    targets = {a: LEG_WEIGHT, b: -LEG_WEIGHT}

    for asset in context.assets:
        order_target_percent(asset, targets.get(asset, 0.0))
