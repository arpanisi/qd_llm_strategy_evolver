"""Step 11: locked baseline evaluation.

Runs the five Step 11 baselines per track over their required window scope:
baseline 2 (equal-weighted daily) runs over train, validation, and test — it
also serves as the Information Ratio benchmark R_b (Step 8A); baselines
1/3/4/5 run over the test window only. Writes the six-metric report per track
to outputs/baselines/<track>/ and persists each window's R_b return series for
Step 8A's IR computations. The two tracks are never combined.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.settings import RunConfig, TrackConfig
from src.engine.metrics import metrics_to_dict
from src.engine.runtime import run_strategy

OUTPUTS = Path(__file__).resolve().parents[2] / "outputs" / "baselines"
BASELINES_DIR = Path(__file__).resolve().parents[1] / "baselines"

IR_BENCHMARK = "baseline_2_equal_weighted"
TEST_ONLY = (
    "baseline_1_market_cap_weighted",
    "baseline_1_equal_notional",
    "baseline_3_risk_parity",
    "baseline_4_macd",
    "baseline_5_rsi_mean_reversion",
)


def load_baselines(track: TrackConfig) -> dict[str, str]:
    base = BASELINES_DIR / track.name
    return {p.stem: p.read_text() for p in sorted(base.glob("baseline_*.py"))}


def _windows(track: TrackConfig) -> list[tuple[str, str, str]]:
    return [
        ("train", str(track.train.start), str(track.train.end)),
        ("validation", str(track.validation.start), str(track.validation.end)),
        ("test", str(track.test.start), str(track.test.end)),
    ]


def evaluate_track(track: TrackConfig, extra: dict | None = None) -> dict:
    baselines = load_baselines(track)
    rows: list[dict] = []
    bench_returns: dict[str, pd.Series] = {}

    for window, start, end in _windows(track):
        if window == "test":
            scope = [IR_BENCHMARK] + [n for n in baselines if n != IR_BENCHMARK]
        else:
            scope = [IR_BENCHMARK]

        for name in scope:
            source = baselines[name]
            bench = bench_returns.get(window)
            perf, metrics, info = run_strategy(
                source, track, start, end, track.starting_cash,
                benchmark_returns=bench, extra=extra,
            )
            if name == IR_BENCHMARK:
                bench_returns[window] = perf["returns"].copy()

            rows.append({
                "track": track.name,
                "baseline": name,
                "window": window,
                "start": start,
                "end": end,
                **metrics_to_dict(metrics),
                "n_fills": metrics.n_fills,
                "n_days": metrics.n_days,
                "n_rejections": info["n_rejections"],
                "final_equity": metrics.final_equity,
            })

    df = pd.DataFrame(rows)
    out = OUTPUTS / track.name
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "baselines.csv", index=False)
    for window, returns in bench_returns.items():
        returns.rename_axis("date").to_csv(out / f"rb_returns_{window}.csv")
    return {"df": df, "bench_returns": bench_returns}


def evaluate_all(cfg: RunConfig) -> dict:
    results: dict[str, dict] = {}
    for track in cfg.tracks:
        extra = None
        if not track.is_futures:
            from src.engine.bundle import equities_market_caps

            extra = {"market_caps": equities_market_caps()}
        results[track.name] = evaluate_track(track, extra=extra)
    return results
