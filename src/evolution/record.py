"""The single strategy record that flows through Steps 5-8.

One record per generated strategy (or per migration copy). It carries the
hypothesis (Step 5), code (Step 6), metrics + status (Step 6), the evaluation
(Step 7), and the selection-relevant fields (Combined Score for Path A,
objectives for Path B). The active path's log (Archive Log / Population
History Log) is a persisted projection of these records.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np

from src.data.taxonomy import StyleVector
from src.roles.schemas import Evaluation, Hypothesis

METRIC_KEYS = (
    "sharpe",
    "sortino",
    "ir",
    "mdd",
    "total_return",
    "turnover",
    "trading_frequency",
    "n_fills",
    "n_days",
    "final_equity",
)


@dataclass
class StrategyRecord:
    strategy_id: str
    track: str
    path: str          # "a" | "b"
    island: int
    generation: int
    hypothesis: Hypothesis
    code: str
    parents: list[str] = field(default_factory=list)
    cousins: list[str] = field(default_factory=list)
    evaluation: Optional[Evaluation] = None
    metrics: dict[str, float] = field(default_factory=dict)
    style: Optional[StyleVector] = None
    status: str = "ok"          # "ok" | "failed"
    failure_error: str = ""

    # Path A: Combined Score == Information Ratio (Step 8A)
    combined_score: float = float("nan")
    # Path B: the four minimization objectives (Step 2B)
    objectives: Optional[np.ndarray] = None

    @property
    def is_failed(self) -> bool:
        return self.status == "failed"

    def metric(self, key: str) -> float:
        return float(self.metrics.get(key, float("nan")))

    def style_bits(self) -> str:
        return self.style.as_binary_string() if self.style else ""

    def to_archive_row(self) -> dict[str, Any]:
        """Flat row for the Path A Archive Log."""
        return {
            "entry_id": self.strategy_id,
            "track": self.track,
            "island": self.island,
            "generation": self.generation,
            "status": self.status,
            "failure_error": self.failure_error,
            "style": self.style_bits(),
            "sharpe": self.metric("sharpe"),
            "sortino": self.metric("sortino"),
            "ir": self.metric("ir"),
            "mdd": self.metric("mdd"),
            "total_return": self.metric("total_return"),
            "turnover": self.metric("turnover"),
            "trading_frequency": self.metric("trading_frequency"),
            "combined_score": self.combined_score,
            "parents": "|".join(self.parents),
            "cousins": "|".join(self.cousins),
            "insight": self.evaluation.insight if self.evaluation else "",
            "style_categories": ",".join(self.evaluation.style_categories) if self.evaluation else "",
        }

    def to_pop_row(self) -> dict[str, Any]:
        """Flat row for the Path B Population History Log."""
        obj = np.zeros(4) if self.objectives is None else np.asarray(self.objectives)
        return {
            "strategy_id": self.strategy_id,
            "track": self.track,
            "island": self.island,
            "generation": self.generation,
            "status": self.status,
            "failure_error": self.failure_error,
            "style": self.style_bits(),
            "f1_neg_sharpe": float(obj[0]),
            "f2_neg_sortino": float(obj[1]),
            "f3_neg_total_return": float(obj[2]),
            "f4_neg_mdd": float(obj[3]),
            "sharpe": self.metric("sharpe"),
            "sortino": self.metric("sortino"),
            "total_return": self.metric("total_return"),
            "mdd": self.metric("mdd"),
            "turnover": self.metric("turnover"),
            "trading_frequency": self.metric("trading_frequency"),
            "parents": "|".join(self.parents),
            "cousins": "|".join(self.cousins),
            "insight": self.evaluation.insight if self.evaluation else "",
            "style_categories": ",".join(self.evaluation.style_categories) if self.evaluation else "",
        }


def metrics_from_dict(data: dict[str, float]) -> dict[str, float]:
    return {k: float(data[k]) for k in METRIC_KEYS if k in data}


def format_strategy_history(record: "StrategyRecord", max_code_chars: int = 3000) -> str:
    """Compact rendering of one strategy's full history (hypothesis, code,
    metrics, prior analysis) for the Step 5/6 prompts."""
    lines = [
        f"--- strategy {record.strategy_id} (island {record.island}, gen {record.generation}, status {record.status}) ---",
        f"style: {record.style_bits()}",
        f"hypothesis: {record.hypothesis.hypothesis}",
        f"rationale: {record.hypothesis.rationale}",
        f"objectives: {record.hypothesis.objectives}",
    ]
    if record.metrics:
        lines.append("metrics (train window): "
                     + ", ".join(f"{k}={record.metric(k):.4f}" for k in
                                 ("sharpe", "sortino", "ir", "mdd", "total_return",
                                  "turnover", "trading_frequency")))
    if record.evaluation is not None:
        ev = record.evaluation
        lines.append(
            f"evaluation: scores h={ev.hypothesis_score:.2f} code={ev.code_alignment_score:.2f} "
            f"results={ev.results_score:.2f}; styles={ev.style_categories}"
        )
        lines.append(f"insight: {ev.insight}")
    if record.status == "failed":
        lines.append(f"FAILED: {record.failure_error}")
    code = record.code.strip()
    if code:
        lines.append(f"code ({len(code)} chars):\n{code[:max_code_chars]}")
    return "\n".join(lines)
