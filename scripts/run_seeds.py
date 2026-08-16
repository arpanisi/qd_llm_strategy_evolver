"""Step 1 entrypoint: backtest all 9 seed strategies per track over that
track's training window and print six-metric reports (IR vs the Step 11
equal-weighted benchmark R_b). Persists outputs/seeds/<track>/seeds.csv.

These seed metrics fix the Path A feature-map bin edges (Step 2A) and the
Path B ideal point (Step 2B).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from src.config.settings import load_config
from src.data.seeds import load_seeds, trading_days_per_month
from src.engine.metrics import metrics_to_dict
from src.engine.runtime import run_strategy

OUTPUTS = ROOT / "outputs" / "seeds"


def _rb_returns(track) -> pd.Series:
    path = ROOT / "outputs" / "baselines" / track.name / "rb_returns_train.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.iloc[:, 0]


def evaluate_track(track) -> pd.DataFrame:
    seeds = load_seeds(track)
    bench = _rb_returns(track)
    extra = {"trading_days_per_month": trading_days_per_month(track)}

    # Step 13: persist each seed's training-window daily returns so PBO can
    # cover seed-occupied archive cells without re-backtesting.
    returns_out = ROOT / "outputs" / "search" / track.name / "returns"
    returns_out.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, source in seeds.items():
        perf, metrics, info = run_strategy(
            source, track,
            str(track.train.start), str(track.train.end),
            track.starting_cash,
            benchmark_returns=bench, extra=extra,
        )
        sid = f"{name}:g0"
        np.savetxt(returns_out / f"{sid}.csv",
                   perf["returns"].astype(float).fillna(0.0).to_numpy())
        rows.append({
            "track": track.name,
            "seed": name,
            "window": "train",
            "start": str(track.train.start),
            "end": str(track.train.end),
            **metrics_to_dict(metrics),
            "n_fills": metrics.n_fills,
            "n_days": metrics.n_days,
            "n_rejections": info["n_rejections"],
            "final_equity": metrics.final_equity,
        })

    df = pd.DataFrame(rows)
    out = OUTPUTS / track.name
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "seeds.csv", index=False)
    return df


def main() -> None:
    cfg = load_config()
    print(f"seed CSVs written to {OUTPUTS}")
    for track in cfg.tracks:
        df = evaluate_track(track)
        cols = [
            "seed", "Sharpe Ratio", "Sortino Ratio", "Information Ratio",
            "Max Drawdown", "Total Return", "Trading Frequency", "Turnover",
            "n_rejections",
        ]
        print(f"\n=== {track.name} (train) ===")
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
