UNIVERSE = ["AAPL", "MSFT", "XOM", "GE", "CVX", "BRK", "PG", "PFE", "JNJ", "WFC", "JPM", "WMT", "BAC", "VZ", "ORCL"]
VOL_LOOKBACK = 20
VOL_SPIKE = 1.5
SPIKE_WEIGHT_MULT = 2.0
VOL_STDEV_WIN = 60
AVG_VOL_WIN = 60

def initialize(context):
    context.assets = [symbol(t) for t in UNIVERSE]

def handle_data(context, data):
    base = 1.0 / len(context.assets)
    raw = {}
    for asset in context.assets:
        v = data.current(asset, "volume")
        h_vol = safe_history(asset, VOL_LOOKBACK + 1, "1d", field="volume")
        avg_v = h_vol.mean() if len(h_vol) >= VOL_LOOKBACK else None
        
        spike = False
        if avg_v and avg_v > 0 and v > VOL_SPIKE * avg_v:
            h_price = safe_history(asset, VOL_STDEV_WIN + AVG_VOL_WIN + 2, "1d", field="close")
            if len(h_price) >= VOL_STDEV_WIN + AVG_VOL_WIN + 1:
                try:
                    rets = h_price.pct_change()
                    vol_s = rets.rolling(VOL_STDEV_WIN).std()
                    avg_vol_s = vol_s.rolling(AVG_VOL_WIN).mean()
                    if len(avg_vol_s) >= 2 and not np.isnan(avg_vol_s.iloc[-1]) and not np.isnan(vol_s.iloc[-1]):
                        if vol_s.iloc[-1] < avg_vol_s.iloc[-1]:
                            spike = True
                except Exception:
                    pass
                    
        raw[asset] = SPIKE_WEIGHT_MULT * base if spike else base
        
    total = sum(raw.values())
    for asset in context.assets:
        order_target_percent(asset, raw[asset] / total)