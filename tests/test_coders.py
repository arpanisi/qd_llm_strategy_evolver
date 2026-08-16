"""Acceptance-criteria tests for Step 6 (coder team loop).

Covers the code-extraction helper, the syntax-check gate, the refinement loop
(error feedback + zero-trade detection), and the failed-after-cap outcome with
the final error preserved. All client/backtest behavior is faked — no live
LLM calls, no Zipline runs.
"""

from __future__ import annotations

import pytest

from src.coders.team import BacktestResult, CoderTeam, _extract_python, write_strategy_code
from src.config.settings import ModelConfig
from src.roles.schemas import Hypothesis


def _models() -> ModelConfig:
    return ModelConfig(
        openrouter_base_url="https://openrouter.ai/api/v1",
        research_model="deepseek/deepseek-v3",
        implementation_model="qwen/qwen2.5-32b-instruct",
        evaluation_model="deepseek/deepseek-v3",
        max_refinement_attempts=2,
        request_timeout_s=30,
        max_retries=1,
    )


def _hypothesis() -> Hypothesis:
    return Hypothesis(
        hypothesis="A testable momentum claim.",
        rationale="Based on parent results.",
        objectives="Hold Sharpe.",
        expected_insights="Something learned either way.",
        risks_limitations="Overfit risk.",
        next_step_ideas="Slow it down.",
    )


class _FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages, model, temperature=None, max_tokens=2048):
        self.calls += 1
        if not self.responses:
            raise RuntimeError("no canned responses left")
        return self.responses.pop(0)


def test_extract_python_fenced_and_raw():
    assert _extract_python("```python\nx = 1\n```") == "x = 1"
    assert _extract_python("here is code:\n\n```py\ny = 2\n```") == "y = 2"
    assert _extract_python("z = 3") == "z = 3"


def test_success_first_attempt(tmp_path):
    client = _FakeClient(["```python\ndef initialize(c): pass\n```"])
    backtests = []
    def backtest(code):
        backtests.append(code)
        return BacktestResult(ok=True, metrics={"n_fills": 10})
    team = CoderTeam(client, _models(), backtest)
    code, result = team.generate("t1", _hypothesis(), parent=_Stub(), cousins=[], track=_Track())
    assert result.ok
    assert len(backtests) == 1
    assert write_strategy_code(tmp_path, "t1", code).exists()


def test_refines_on_error_then_succeeds():
    client = _FakeClient([
        "```python\ndef initialize(c): raise ValueError('boom')\n```",
        "```python\ndef initialize(c): pass\n```",
    ])
    calls = {"n": 0}
    def backtest(code):
        calls["n"] += 1
        if calls["n"] == 1:
            return BacktestResult(ok=False, error="ValueError: boom")
        return BacktestResult(ok=True, metrics={"n_fills": 5})
    team = CoderTeam(client, _models(), backtest)
    code, result = team.generate("t2", _hypothesis(), parent=_Stub(), cousins=[], track=_Track())
    assert result.ok
    assert calls["n"] == 2


def test_refines_on_zero_trades():
    client = _FakeClient([
        "```python\ndef initialize(c): pass\n```",
        "```python\ndef initialize(c): pass\n```",
        "```python\ndef initialize(c): pass\n```",
    ])
    calls = {"n": 0}
    def backtest(code):
        calls["n"] += 1
        return BacktestResult(ok=True, metrics={"n_fills": 0})
    team = CoderTeam(client, _models(), backtest)
    code, result = team.generate("t3", _hypothesis(), parent=_Stub(), cousins=[], track=_Track())
    assert not result.ok
    assert calls["n"] == 3  # 1 initial + 2 refinements (max_refinement_attempts)
    assert "zero trades" in result.error


def test_syntax_error_does_not_consume_backtest():
    client = _FakeClient([
        "```python\ndef initialize(c): ::\n```",
        "```python\ndef initialize(c): pass\n```",
    ])
    calls = {"n": 0}
    def backtest(code):
        calls["n"] += 1
        return BacktestResult(ok=True, metrics={"n_fills": 3})
    team = CoderTeam(client, _models(), backtest)
    _, result = team.generate("t4", _hypothesis(), parent=_Stub(), cousins=[], track=_Track())
    assert result.ok
    assert calls["n"] == 1  # syntax error never reached the backtest


class _Stub:
    strategy_id = "seed_1:g0"
    island = 1
    generation = 0
    status = "ok"
    failure_error = ""
    style = None
    metrics = {}
    evaluation = None
    code = "def initialize(c): pass"

    def metric(self, key): return 0.0
    def style_bits(self): return "00000000"
    @property
    def hypothesis(self):
        return _hypothesis()


class _Track:
    name = "equities"
    is_futures = False
    train = type("R", (), {"start": "2015-01-01", "end": "2020-12-31"})()
    validation = type("R", (), {"start": "2021-01-01", "end": "2022-12-31"})()
    test = type("R", (), {"start": "2023-01-01", "end": "2024-12-31"})()
    starting_cash = 1_000_000.0
    bars_per_year = 252
