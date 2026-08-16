"""Steps 5/7 schemas and pure validators.

The validators are deliberately free of any LLM call so they can be unit-tested
directly. The role classes (research.py / evaluate.py) use them to enforce the
hard schema requirements of Step 5 (all six hypothesis fields, none empty, none
a restatement of another) and Step 7 (all eight evaluation fields, scores in
[0, 1], style tags from the locked taxonomy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.data.taxonomy import STYLE_NAMES, N_STYLES

HYPOTHESIS_FIELDS = (
    "hypothesis",
    "rationale",
    "objectives",
    "expected_insights",
    "risks_limitations",
    "next_step_ideas",
)

EVALUATION_FIELDS = (
    "hypothesis_score",
    "hypothesis_reasoning",
    "code_alignment_score",
    "code_reasoning",
    "results_score",
    "results_reasoning",
    "style_categories",
    "insight",
)

# narrative fields that must be more than a token placeholder
_MIN_NARRATIVE = 12


@dataclass
class Hypothesis:
    hypothesis: str
    rationale: str
    objectives: str
    expected_insights: str
    risks_limitations: str
    next_step_ideas: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Hypothesis":
        return cls(**{f: str(data.get(f, "") or "") for f in HYPOTHESIS_FIELDS})

    def to_dict(self) -> dict[str, str]:
        return {f: getattr(self, f) for f in HYPOTHESIS_FIELDS}


@dataclass
class Evaluation:
    hypothesis_score: float
    hypothesis_reasoning: str
    code_alignment_score: float
    code_reasoning: str
    results_score: float
    results_reasoning: str
    style_categories: list[str]
    insight: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Evaluation":
        cats = data.get("style_categories", [])
        if not isinstance(cats, list):
            cats = []
        return cls(
            hypothesis_score=_as_score(data.get("hypothesis_score")),
            hypothesis_reasoning=str(data.get("hypothesis_reasoning", "") or ""),
            code_alignment_score=_as_score(data.get("code_alignment_score")),
            code_reasoning=str(data.get("code_reasoning", "") or ""),
            results_score=_as_score(data.get("results_score")),
            results_reasoning=str(data.get("results_reasoning", "") or ""),
            style_categories=[str(c) for c in cats],
            insight=str(data.get("insight", "") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_score": self.hypothesis_score,
            "hypothesis_reasoning": self.hypothesis_reasoning,
            "code_alignment_score": self.code_alignment_score,
            "code_reasoning": self.code_reasoning,
            "results_score": self.results_score,
            "results_reasoning": self.results_reasoning,
            "style_categories": list(self.style_categories),
            "insight": self.insight,
        }


def _as_score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _overlap_ratio(a: str, b: str) -> float:
    """Heuristic pairwise-distinctness check: normalized token overlap."""
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def validate_hypothesis(h: Hypothesis) -> tuple[bool, list[str]]:
    """Step 5 hard schema check: six fields, all populated, all genuinely
    distinct content (no empty or restated fields)."""
    errors: list[str] = []
    values = {f: getattr(h, f).strip() for f in HYPOTHESIS_FIELDS}
    for f in HYPOTHESIS_FIELDS:
        if not values[f]:
            errors.append(f"missing required field: {f}")
        elif f in ("hypothesis", "rationale", "expected_insights") and len(values[f]) < _MIN_NARRATIVE:
            errors.append(f"{f} is too short to be a real claim ({len(values[f])} chars)")
    # distinctness: any pair of narrative fields with near-total token overlap
    for i, f1 in enumerate(HYPOTHESIS_FIELDS):
        for f2 in HYPOTHESIS_FIELDS[i + 1 :]:
            if values[f1] and values[f2] and _overlap_ratio(values[f1], values[f2]) > 0.8:
                errors.append(f"{f1} is a restatement of {f2}")
    return (len(errors) == 0, errors)


def validate_evaluation(e: Evaluation) -> tuple[bool, list[str]]:
    """Step 7 schema check: scores in [0,1], taxonomy style tags, non-empty
    reasoning/insight."""
    errors: list[str] = []
    for f in ("hypothesis_score", "code_alignment_score", "results_score"):
        v = getattr(e, f)
        if v != v or not (0.0 <= v <= 1.0):
            errors.append(f"{f} must be a number in [0, 1], got {v!r}")
    for f in ("hypothesis_reasoning", "code_reasoning", "results_reasoning", "insight"):
        if not getattr(e, f).strip():
            errors.append(f"missing required field: {f}")
    valid_styles = set(STYLE_NAMES)
    for cat in e.style_categories:
        if cat not in valid_styles:
            errors.append(f"style_categories has unknown category: {cat!r}")
    return (len(errors) == 0, errors)
