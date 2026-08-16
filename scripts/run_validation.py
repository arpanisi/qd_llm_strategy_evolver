"""Step 13 entrypoint: multiple-testing-corrected significance diagnostics for
a completed tier, per track — PBO via CSCV (both paths) and Deflated Sharpe
Ratio (Path A). Read-only over persisted data; no re-backtest.

Usage:
    python scripts/run_validation.py [tier]     # default tier3
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.report.backtest_validation import run_validation  # noqa: E402


def main() -> None:
    tier = sys.argv[1] if len(sys.argv) > 1 else "tier3"
    for track in ("equities", "futures"):
        report = run_validation(track, tier)
        pa, pb = report["path_a"], report["path_b"]
        fmt = lambda x: "undefined" if x is None else f"{x:.4f}"
        print(f"=== {track} ({tier}) ===")
        print(f"  Path A: PBO={fmt(pa['pbo']['pbo'])}  "
              f"DSR={fmt(pa['dsr']['dsr'])}")
        print(f"  Path B: PBO={fmt(pb['pbo']['pbo'])}")


if __name__ == "__main__":
    main()
