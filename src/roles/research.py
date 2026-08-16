"""Step 5 — Research Role: Hypothesis Generation.

Given the parent + 7 cousins (full history: hypotheses, code, metrics, prior
analysis) and the current Insight Log, produces a structured hypothesis object
with exactly the six Step 5 fields, all populated with genuinely distinct
content. The hard schema requirement is enforced by validate_hypothesis();
the model is re-requested with the specific failing fields until it complies
(a bounded loop) or the candidate is marked failed.
"""

from __future__ import annotations

import logging

from src.config.settings import ModelConfig, TrackConfig
from src.evolution.record import StrategyRecord, format_strategy_history
from src.llm.client import OpenRouterClient
from src.roles.schemas import Hypothesis, HYPOTHESIS_FIELDS, validate_hypothesis

log = logging.getLogger("roles.research")

SYSTEM_PROMPT = (
    "You are a quantitative strategy research analyst. You propose testable "
    "trading hypotheses for an evolutionary strategy-discovery system. Each "
    "hypothesis must be specific, falsifiable, and conditioned on the "
    "parent/cousin strategies and the accumulated insights provided. "
    "Return strictly a JSON object with exactly these six keys: "
    + ", ".join(HYPOTHESIS_FIELDS)
    + ". Every field must be non-empty and materially distinct from the "
    "others. Keep the hypothesis to 1-2 sentences."
)


def _correction_prompt(errors: list[str]) -> str:
    detail = "; ".join(errors)
    return (
        f"The previous hypothesis was rejected for the following reason(s): {detail}. "
        f"Return a corrected JSON object with all six fields: "
        + ", ".join(HYPOTHESIS_FIELDS) + "."
    )


class ResearchRole:
    def __init__(self, client: OpenRouterClient, models: ModelConfig) -> None:
        self.client = client
        self.model = models.research_model
        self.max_attempts = models.max_refinement_attempts

    def generate_hypothesis(
        self,
        parent: StrategyRecord,
        cousins: list[StrategyRecord],
        insights: list[str],
        track: TrackConfig,
    ) -> Hypothesis:
        history = "\n\n".join(
            [format_strategy_history(parent)]
            + [format_strategy_history(c) for c in cousins]
        )
        insight_block = (
            "\n".join(f"- {i}" for i in insights[-25:])
            if insights
            else "(no accumulated insights yet)"
        )
        user = (
            f"Track: {track.name}. Training window {track.train.start}..{track.train.end}. "
            f"Universe: "
            + ("15 equities" if not track.is_futures else "ES and NQ futures")
            + ".\n\nParent and cousins:\n"
            + history
            + "\n\nIsland Insight Log (latest):\n"
            + insight_block
            + "\n\nPropose a hypothesis JSON object with all six fields."
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.append({"role": "user", "content": user})

        for attempt in range(self.max_attempts):
            data = self.client.chat_json(messages, self.model)
            hypothesis = Hypothesis.from_dict(data)
            ok, errors = validate_hypothesis(hypothesis)
            if ok:
                return hypothesis
            log.info("hypothesis rejected (attempt %d): %s", attempt + 1, errors)
            messages.append({"role": "assistant", "content": self.client.chat(messages, self.model)})
            messages.append({"role": "user", "content": _correction_prompt(errors)})
        raise RuntimeError(
            f"research role failed to produce a valid hypothesis after "
            f"{self.max_attempts} attempts"
        )
