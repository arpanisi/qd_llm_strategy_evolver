"""Step 7 — Evaluation Role.

Scores a generated strategy on three axes (hypothesis soundness, code/hypothesis
alignment, empirical results) plus the 8-bit style-category assignment and a
candidate insight for the island Insight Log. The hard Step 7 schema
requirements are enforced by validate_evaluation(); non-compliant responses are
re-requested in a bounded loop before the candidate is marked failed.
"""

from __future__ import annotations

import logging

from src.config.settings import ModelConfig
from src.data.taxonomy import STYLE_NAMES
from src.evolution.record import StrategyRecord
from src.llm.client import OpenRouterClient
from src.roles.schemas import (
    EVALUATION_FIELDS,
    Evaluation,
    validate_evaluation,
)

log = logging.getLogger("roles.evaluate")

SYSTEM_PROMPT = (
    "You are a rigorous quant backtest reviewer. You evaluate one generated "
    "strategy per session on three axes, each a score in [0, 1] with a "
    "one-sentence justification, and you tag which of the locked strategy "
    "styles the result belongs to. Return strictly a JSON object with exactly "
    "these keys: "
    + ", ".join(EVALUATION_FIELDS)
    + f". style_categories must be a JSON array of names drawn ONLY from: "
    + ", ".join(STYLE_NAMES)
    + ". Be skeptical of overfit signals."
)


class EvaluationRole:
    def __init__(self, client: OpenRouterClient, models: ModelConfig) -> None:
        self.client = client
        self.model = models.evaluation_model
        self.max_attempts = models.max_refinement_attempts

    def evaluate(
        self,
        record: StrategyRecord,
        backtest_log: str,
        baseline_summary: str,
    ) -> Evaluation:
        metrics = ", ".join(
            f"{k}={record.metric(k):.4f}"
            for k in ("sharpe", "sortino", "ir", "mdd", "total_return",
                      "turnover", "trading_frequency")
        )
        user = (
            f"Strategy {record.strategy_id} (island {record.island}, gen {record.generation}).\n"
            f"Hypothesis: {record.hypothesis.hypothesis}\n"
            f"Rationale: {record.hypothesis.rationale}\n"
            f"Expected insights: {record.hypothesis.expected_insights}\n"
            f"Style tag: {record.style_bits()}\n\n"
            f"Code ({len(record.code)} chars):\n{record.code[:4000]}\n\n"
            f"Train metrics: {metrics}\n"
            f"IR benchmark (equal-weight buy-and-hold) summary: {baseline_summary}\n"
            f"Backtest log tail:\n{backtest_log[-3000:]}\n\n"
            f"Return the evaluation JSON object."
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.append({"role": "user", "content": user})

        for attempt in range(self.max_attempts):
            data = self.client.chat_json(messages, self.model)
            evaluation = Evaluation.from_dict(data)
            ok, errors = validate_evaluation(evaluation)
            if ok:
                return evaluation
            log.info("evaluation rejected (attempt %d): %s", attempt + 1, errors)
            messages.append({"role": "assistant", "content": self.client.chat(messages, self.model)})
            messages.append({
                "role": "user",
                "content": "The previous evaluation was rejected: "
                           + "; ".join(errors)
                           + ". Return a corrected JSON object.",
            })
        raise RuntimeError(
            f"evaluation role failed to produce a valid evaluation after "
            f"{self.max_attempts} attempts"
        )
