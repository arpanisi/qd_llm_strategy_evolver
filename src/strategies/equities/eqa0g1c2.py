UNIVERSE = ["AAPL", "MSFT", "XOM", "GE", "CVX", "BRK", "PG", "PFE", "JNJ", "WFC", "JPM", "WMT", "BAC", "VZ", "ORCL"]
TOP_N = 3
WARMUP = 91
DD_LIMIT = -0.15

def initialize(context):
    context.assets = [symbol(t) for t in UNIVERSE]
    schedule_function(rebalance, date_rules.month_start(), time_rules.market_open())

def rebalance(context, data):
    pool = {}
    for asset in context.assets:
        hist = safe_history(asset, WARMUP, "1d")
        if len(hist) >= WARMUP:
            hist = hist.dropna()
            if len(hist) >= WARMUP:
                ret = hist.iloc[-1] / hist.iloc[0] - 1.0
                dd = (hist / hist.cummax()) - 1.0
                if dd.min() >= DD_LIMIT:
                    pool[asset] = ret
    top = sorted(pool, key=pool.get, reverse=True)[:TOP_N]
    w = 1.0 / TOP_N if len(top) == TOP_N else 0.0
    for a in context.assets:
        order_target_percent(a, w if a in top else 0.0)