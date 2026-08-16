"""Acceptance-criteria tests for Steps 5/7 (schemas, research/evaluation
validators) and the Step 7 Insight Log + curation."""

from __future__ import annotations

import numpy as np
import pytest

from src.roles.insights import InsightLog
from src.roles.schemas import (
    Evaluation,
    Hypothesis,
    validate_evaluation,
    validate_hypothesis,
)


def _good_hypothesis() -> Hypothesis:
    return Hypothesis(
        hypothesis="12-month momentum conditioned on low realized volatility "
                   "outperforms unconditional momentum in this universe.",
        rationale="Parent's momentum signal showed a clear drawdown cliff in "
                  "the 2018 vol spike; conditioning on realized volatility "
                  "directly addresses that observed failure mode.",
        objectives="Reduce max drawdown below parent's -35% while keeping "
                   "Sharpe above 1.0.",
        expected_insights="If confirmed, volatility-filtered momentum is a "
                          "real improvement; if refuted, the observed cliff "
                          "was not driven by volatility.",
        risks_limitations="Regime-dependence risk and overfitting to the "
                          "2015-2020 sample window.",
        next_step_ideas="Try a slower vol-scaled window, then a long-only "
                        "variant.",
    )


def test_hypothesis_six_fields_all_valid():
    ok, errors = validate_hypothesis(_good_hypothesis())
    assert ok, errors


def test_hypothesis_missing_field_rejected():
    h = _good_hypothesis()
    h.risks_limitations = ""
    ok, errors = validate_hypothesis(h)
    assert not ok
    assert any("risks_limitations" in e for e in errors)


def test_hypothesis_restated_field_rejected():
    h = _good_hypothesis()
    h.next_step_ideas = h.hypothesis
    ok, errors = validate_hypothesis(h)
    assert not ok
    assert any("restatement" in e for e in errors)


def _good_evaluation() -> Evaluation:
    return Evaluation(
        hypothesis_score=0.8, hypothesis_reasoning="Clear and testable claim "
        "grounded in the parent's observed failure.",
        code_alignment_score=0.9, code_reasoning="Implements the stated vol "
        "filter exactly.",
        results_score=0.6, results_reasoning="Sharpe improved, drawdown "
        "reduced as predicted but turnover rose.",
        style_categories=["trend_following", "volatility"],
        insight="Volatility-scaled position sizing reduced drawdown by 8pp "
                "but cost 0.15 Sharpe.",
    )


def test_evaluation_all_fields_valid():
    ok, errors = validate_evaluation(_good_evaluation())
    assert ok, errors


def test_evaluation_score_out_of_range_rejected():
    e = _good_evaluation()
    e.hypothesis_score = 1.5
    ok, errors = validate_evaluation(e)
    assert not ok
    assert any("hypothesis_score" in err for err in errors)


def test_evaluation_unknown_style_rejected():
    e = _good_evaluation()
    e.style_categories = ["not_a_real_style"]
    ok, errors = validate_evaluation(e)
    assert not ok
    assert any("style_categories" in err for err in errors)


def test_evaluation_from_dict_coerces_scores():
    e = Evaluation.from_dict({
        "hypothesis_score": "0.75",
        "hypothesis_reasoning": "ok",
        "code_alignment_score": 1.0,
        "code_reasoning": "ok",
        "results_score": "0.5",
        "results_reasoning": "ok",
        "style_categories": ["mean_reversion"],
        "insight": "insight",
    })
    assert e.hypothesis_score == pytest.approx(0.75)
    assert e.style_categories == ["mean_reversion"]


# ---------------------------------------------------------------------------
# Insight Log (Step 7): dedup + K curation
# ---------------------------------------------------------------------------

def test_insight_log_dedups_and_caps(tmp_path):
    log = InsightLog(tmp_path / "insights.csv", k=2)
    log.add("Volatility scaling cut drawdown but cost Sharpe.", 0.9, "s1", 1)
    log.add("Volatility scaling cut drawdown but cost Sharpe (variant).", 0.95, "s2", 2)
    log.add("Calendar seasonality works in Jan and Mar.", 0.5, "s3", 2)
    assert len(log.entries) <= 2
    assert log.entries[0]["insight"].startswith("Volatility scaling")


def test_insight_log_curation_merges_and_trims(tmp_path):
    log = InsightLog(tmp_path / "insights.csv", k=2)
    log.add("A long lesson one.", 0.4, "s1", 1)
    log.add("A long lesson two, different.", 0.3, "s2", 1)
    log.add("A long lesson one, slightly reworded.", 0.8, "s3", 1)
    log.curate()
    assert len(log.entries) <= 2
    assert log.entries[0]["score"] == pytest.approx(0.8)


def test_insight_log_persists(tmp_path):
    log = InsightLog(tmp_path / "insights.csv", k=3)
    log.add("Insight X.", 0.7, "s1", 1)
    reloaded = InsightLog(tmp_path / "insights.csv", k=3)
    assert reloaded.recent() == ["Insight X."]
