"""Step 12 — Path Comparison report, per track (never across tracks).

Loads each path's run summary, computes average Turnover across the final
strategy sets, and runs the decisive dominance check on test-window metrics:
for every Path A final strategy, is any Path B Pareto member at least as good
on all of {Sharpe, Sortino, Total Return, Max Drawdown} and strictly better on
at least one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

OUTPUTS = Path(__file__).resolve().parents[2] / "outputs" / "evolution"
REPORT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "reports"

_TEST_OBJECTIVES = ("sharpe", "sortino", "total_return", "mdd")


def _path_summary(track: str, path: str, tier: str) -> dict:
    p = OUTPUTS / track / path / tier / "summary.json"
    with open(p) as fh:
        return json.load(fh)


def average_turnover(summary: dict, path: str) -> float:
    """Mean test-window Turnover across the path's final strategy set."""
    turnovers = []
    if path == "a":
        for island in summary["best_per_island"].values():
            t = island.get("test", {}).get("turnover")
            if t is not None and np.isfinite(t):
                turnovers.append(float(t))
    else:
        for sid in summary["test"]:
            t = summary["test"][sid].get("turnover")
            if t is not None and np.isfinite(t):
                turnovers.append(float(t))
    return float(np.mean(turnovers)) if turnovers else float("nan")


def _test_vector(metrics: dict) -> Optional[np.ndarray]:
    try:
        v = np.array([float(metrics[k]) for k in _TEST_OBJECTIVES])
    except (KeyError, TypeError, ValueError):
        return None
    return v if np.all(np.isfinite(v)) else None


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True if a dominates b on all four objectives (at least as good on all,
    strictly better on at least one) — all in 'higher is better' form."""
    return bool(np.all(a >= b) and np.any(a > b))


def compare_track(track: str, tier: str) -> dict:
    sa = _path_summary(track, "a", tier)
    sb = _path_summary(track, "b", tier)

    path_a_finals = {
        island: island_info["test"]
        for island, island_info in sa["best_per_island"].items()
    }
    path_b_pareto = {
        sid: m for sid, m in sb["test"].items()
    }

    dominated: list[dict] = []
    for island, metrics in path_a_finals.items():
        va = _test_vector(metrics)
        if va is None:
            continue
        for sid, bm in path_b_pareto.items():
            vb = _test_vector(bm)
            if vb is not None and dominates(vb, va):
                dominated.append({
                    "path_a_island": island,
                    "path_a_entry": sa["best_per_island"][island]["entry_id"],
                    "dominated_by": sid,
                    "path_b_better_on": [
                        k for k in _TEST_OBJECTIVES
                        if vb[_TEST_OBJECTIVES.index(k)] > va[_TEST_OBJECTIVES.index(k)]
                    ],
                })

    report = {
        "track": track,
        "tier": tier,
        "path_a": {
            "breadth": sa["breadth"]["breadth"],
            "style_distribution": sa["breadth"]["distribution"],
            "avg_test_turnover": average_turnover(sa, "a"),
            "n_final": len(path_a_finals),
        },
        "path_b": {
            "breadth": sb["breadth"]["breadth"],
            "style_distribution": sb["breadth"]["distribution"],
            "avg_test_turnover": average_turnover(sb, "b"),
            "n_pareto": len(path_b_pareto),
        },
        "decisive_check": {
            "n_path_a_strategies_dominated": len(dominated),
            "dominated_path_a_strategies": dominated,
            "conclusion": (
                "Path B found nothing that dominates any Path A final strategy "
                "(no evidence Path A's scoring costs anything on this universe)"
                if not dominated else
                "Path B captured value Path A's scoring could not reach on "
                f"{len(dominated)} Path A final strategy(ies)"
            ),
        },
    }
    return report


def write_report(track: str, tier: str) -> dict:
    report = compare_track(track, tier)
    out = REPORT_DIR / track
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"path_comparison_{tier}.json", "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    return report
