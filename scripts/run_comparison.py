"""Step 12 entrypoint: produce the per-track Path A vs Path B comparison report
from a completed tier's summaries (defaults to tier2, the first tier with real
reportable numbers)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.report.comparison import write_report  # noqa: E402


def main() -> None:
    tier = sys.argv[1] if len(sys.argv) > 1 else "tier2"
    for track in ("equities", "futures"):
        report = write_report(track, tier)
        print(f"\n=== {track} ({tier}) ===")
        print(f"  Path A breadth={report['path_a']['breadth']:.3f}  "
              f"avg test turnover={report['path_a']['avg_test_turnover']:.4f}")
        print(f"  Path B breadth={report['path_b']['breadth']:.3f}  "
              f"avg test turnover={report['path_b']['avg_test_turnover']:.4f}")
        dc = report["decisive_check"]
        print(f"  dominated Path A finals: {dc['n_path_a_strategies_dominated']} — {dc['conclusion']}")


if __name__ == "__main__":
    main()
