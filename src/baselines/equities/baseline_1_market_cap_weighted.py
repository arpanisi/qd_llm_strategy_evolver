"""Baseline 1 (equities): market-cap-weighted, rebalanced monthly.

Weights = dlycap_i / sum(dlycap) over the universe at each monthly rebalance,
read from the injected `market_caps` series (sid -> daily CRSP market cap),
which the runner preloads from the Step 1 parquet. Test-window-only baseline
(Step 11) — never used as the Information Ratio benchmark and never computed
for train/validation.
"""

UNIVERSE = [
    "AAPL", "MSFT", "XOM", "GE", "CVX", "BRK", "PG", "PFE",
    "JNJ", "WFC", "JPM", "WMT", "BAC", "VZ", "ORCL",
]


def initialize(context):
    context.assets = [symbol(t) for t in UNIVERSE]
    schedule_function(rebalance, date_rules.month_start(), time_rules.market_open())


def rebalance(context, data):
    ts = pd.Timestamp(data.current_dt).tz_localize(None).normalize()
    caps = {}
    total = 0.0
    for asset in context.assets:
        cap = market_caps[asset.sid].asof(ts)
        cap = 0.0 if (cap != cap) else float(cap)
        caps[asset] = cap
        total += cap
    if total <= 0.0:
        return
    for asset in context.assets:
        order_target_percent(asset, caps[asset] / total)
