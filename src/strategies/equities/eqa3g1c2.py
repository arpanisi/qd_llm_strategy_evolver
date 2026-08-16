UNIVERSE = ["AAPL", "MSFT", "XOM", "GE", "CVX", "BRK", "PG", "PFE", "JNJ", "WFC", "JPM", "WMT", "BAC", "VZ", "ORCL"]
VOL_LOOKBACK = 20
VOL_SPIKE = 1.5
SPIKE_WEIGHT_MULT = 2.0

def initialize(context):
    context.assets = [symbol(t) for t in UNIVERSE]

def handle_data(context, data):
    base = 1.0 / len(context.assets)
    raw = {}
    for asset in context.assets:
        v = data.current(asset, "volume")
        hist = safe_history(asset, VOL_LOOKBACK + 1, "1d", field="volume")
        avg_v = hist.mean() if len(hist) >= VOL_LOOKBACK else None
        
        prices = safe_history(asset, VOL_LOOKBACK + 1, "1d", field="close")
        spike = False
        if avg_v and avg_v > 0 and v > VOL_SPIKE * avg_v:
            if len(prices) >= VOL_LOOKBACK + 1:
                rets = prices.pct_change().dropna()
                if len(rets) >= VOL_LOOKBACK:
                    vols = rets.rolling(10).std()
                    cv = vols.iloc[-1]
                    av = vols.rolling(20).mean().iloc[-1]
                    if not pd.isna(cv) and not pd.isna(av) and cv < av:
                        spike = True
                        
        raw[asset] = SPIKE_WEIGHT_MULT * base if spike else base
        
    total = sum(raw.values())
    for asset in context.assets:
        order_target_percent(asset, raw[asset] / total)