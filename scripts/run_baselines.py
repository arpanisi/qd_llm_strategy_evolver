"""Step 11 entrypoint: run the five locked baselines per track and print the
six-metric reports. Baseline 2 (equal-weighted daily) runs over train /
validation / test and is persisted as the IR benchmark R_b per window;
baselines 1/3/4/5 run over the test window only.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.settings import load_config
from src.eval.baselines import OUTPUTS, evaluate_all


def main() -> None:
    cfg = load_config()
    results = evaluate_all(cfg)

    print(f"baseline CSVs written to {OUTPUTS}")
    for track in cfg.tracks:
        df = results[track.name]["df"]
        cols = [
            "baseline", "window", "Sharpe Ratio", "Sortino Ratio",
            "Information Ratio", "Max Drawdown", "Total Return",
            "Trading Frequency", "Turnover", "n_rejections",
        ]
        print(f"\n=== {track.name} ===")
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
