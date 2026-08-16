"""Debug: run a single seed and print rejection context (gross, pv, position)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.engine.runtime as rt
from src.config.settings import load_config
from src.data.seeds import load_seeds, trading_days_per_month

real_log = rt._log_rejection


def debug_log(order_fn, asset, reason):
    ctx = rt._CTX.get("context")
    data = rt._CTX.get("data")
    gross = rt._current_gross_exposure(ctx) if ctx else None
    pv = ctx.portfolio.portfolio_value if ctx else None
    price = data.current(asset, "close") if data else None
    pos = ctx.portfolio.positions.get(asset) if ctx else None
    amount = pos.amount if pos else 0.0
    last = pos.last_sale_price if pos else 0.0
    print(
        f"REJECT {order_fn} {getattr(asset, 'symbol', asset)} "
        f"pv={pv:.0f} gross={gross:.0f} price={price} amount={amount} last={last}"
    )
    return real_log(order_fn, asset, reason)


rt._log_rejection = debug_log

if __name__ == "__main__":
    which = sys.argv[1]
    track_name = sys.argv[2]
    cfg = load_config()
    track = cfg.track(track_name)
    seeds = load_seeds(track)
    source = seeds[which]
    bench = None
    extra = {"trading_days_per_month": trading_days_per_month(track)}
    perf, metrics, info = rt.run_strategy(
        source, track, str(track.train.start), str(track.train.end),
        track.starting_cash, benchmark_returns=bench, extra=extra,
    )
    print("rejections:", info["n_rejections"], "n_fills:", metrics.n_fills)
