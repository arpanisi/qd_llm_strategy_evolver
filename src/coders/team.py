"""Step 6 — Implementation Team: initial implementation → backtest → refine.

One generated strategy per call. The model first translates the Step 5
hypothesis into Zipline code; the injected ``backtest`` executor runs it over
the training window under the real cost model. On a crash, a syntax error, or
a "clearly broken" result (zero trades ever), the loop feeds back the original
hypothesis, the original code, and the exact error/traceback and requests a
corrected version — capped at ``max_refinement_attempts`` refinements. A
strategy that still fails is marked failed with its final error preserved; it
is excluded from Step 7 and from archive/population placement.
"""

from __future__ import annotations

import logging
import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from src.config.settings import ModelConfig, TrackConfig
from src.evolution.record import StrategyRecord, format_strategy_history
from src.llm.client import OpenRouterClient
from src.roles.schemas import Hypothesis

log = logging.getLogger("coders.team")

SYSTEM_PROMPT = """You are a Zipline strategy engineer. You write complete, self-contained
strategy modules. Output ONLY a python code block between ```python and ```.

The module is executed in a namespace that provides:
- `initialize(context)` (required) and `handle_data(context, data)` (optional)
- `symbol(ticker)` for equities; for futures use `future_symbol("ES")` /
  `future_symbol("NQ")` (the futures symbols; `symbol()` is not available on
  the futures track)
- order functions: `order(asset, amount)`, `order_target_percent(asset, pct)`,
  `order_target_value`, `order_value`, `order_percent`, `order_target`
  - Equities track: `order_target_percent`/`order_percent` are fine.
  - Futures track: use `order(asset, n_contracts)` with integer contract counts
    ONLY (percent/value variants raise NotImplementedError there).
- `safe_history(asset, bar_count, "1d")` for close history (tolerates a too-
  short warmup: returns an empty Series, so guard with len()).
- `BARS_PER_YEAR`, `np`, `pd`, `math`, plus all standard zipline.api helpers
  (`schedule_function`, `record`, `get_open_orders`, `get_datetime`, ...).
- Never assume a symbol was added by the plan: build the universe yourself in
  initialize from the known instrument list described in the user prompt.
- No lookahead: use only data at or before the current bar.
- Guard every access so a missing/gap bar cannot crash the run; wrap risky
  calls and default to holding the prior position or going flat.
Keep the code short (under ~120 lines), deterministic, and free of comments
that restate the code."""


@dataclass
class BacktestResult:
    ok: bool
    metrics: dict[str, float] = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    error: str = ""
    code: str = ""

    def broken_reason(self) -> Optional[str]:
        """'Clearly broken' trigger per Step 6 stage 3: zero trades ever."""
        if self.ok and int(self.metrics.get("n_fills", 0)) == 0:
            return "the backtest completed but made zero trades over the entire training window"
        return None


class CoderTeam:
    def __init__(
        self,
        client: OpenRouterClient,
        models: ModelConfig,
        backtest: Callable[[str], BacktestResult],
    ) -> None:
        self.client = client
        self.model = models.implementation_model
        self.max_attempts = models.max_refinement_attempts
        self.backtest = backtest

    def _completion(self, messages: list[dict]) -> str:
        text = self.client.chat(messages, self.model, temperature=0.2, max_tokens=4096)
        return _extract_python(text)

    def generate(
        self,
        strategy_id: str,
        hypothesis: Hypothesis,
        parent: StrategyRecord,
        cousins: list[StrategyRecord],
        track: TrackConfig,
    ) -> tuple[str, BacktestResult]:
        history = "\n\n".join(
            [format_strategy_history(parent)]
            + [format_strategy_history(c) for c in cousins]
        )
        universe_line = (
            "Universe: AAPL, MSFT, XOM, GE, CVX, BRK, PG, PFE, JNJ, WFC, JPM, WMT, BAC, VZ, ORCL"
            if not track.is_futures
            else "Universe: ES and NQ futures (multipliers ES=50, NQ=20; price in points; "
                 "positions are integer contracts)"
        )
        user = (
            f"Strategy id: {strategy_id}\nTrack: {track.name}. Training window "
            f"{track.train.start}..{track.train.end}. {universe_line}\n"
            f"Style tag: {parent.style_bits() if parent.style else 'n/a'}\n\n"
            f"Hypothesis to implement:\n{hypothesis.hypothesis}\n"
            f"Rationale: {hypothesis.rationale}\n"
            f"Objectives: {hypothesis.objectives}\n\n"
            f"Parent and cousins (for conditioning):\n{history[:6000]}\n\n"
            f"Write the strategy module."
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.append({"role": "user", "content": user})

        first_code = None
        for attempt in range(self.max_attempts + 1):
            code = self._completion(messages)
            if first_code is None:
                first_code = code
            if not code.strip():
                messages.append({
                    "role": "assistant",
                    "content": "(no code emitted)",
                })
                messages.append({
                    "role": "user",
                    "content": "You returned no python code. Output ONLY a ```python``` block.",
                })
                continue
            try:
                compile(code, f"<{strategy_id}>", "exec")
            except SyntaxError as exc:
                messages.append({"role": "assistant", "content": code})
                messages.append({
                    "role": "user",
                    "content": _refine_prompt(hypothesis, code, f"SyntaxError: {exc}"),
                })
                continue

            result = self.backtest(code)
            result.code = code
            broken = result.broken_reason() if result.ok else None
            if result.ok and broken is None:
                log.info("%s implemented cleanly on attempt %d", strategy_id, attempt + 1)
                return code, result
            detail = result.error if not result.ok else broken
            if attempt < self.max_attempts:
                messages.append({"role": "assistant", "content": code})
                messages.append({
                    "role": "user",
                    "content": _refine_prompt(hypothesis, code, detail),
                })
            else:
                final = BacktestResult(
                    ok=False,
                    metrics=result.metrics,
                    info=result.info,
                    error=result.error or str(broken),
                    code=code,
                )
                log.warning("%s failed after %d refinement attempts: %s",
                            strategy_id, self.max_attempts, detail[:300])
                return code, final

        # unreachable
        raise RuntimeError(f"coder loop exhausted for {strategy_id}")


def _refine_prompt(hypothesis: Hypothesis, code: str, problem: str) -> str:
    return (
        "The implementation was rejected. Fix it and output ONLY a new ```python``` block.\n\n"
        f"Original hypothesis:\n{hypothesis.hypothesis}\n"
        f"Original code:\n```python\n{code}\n```\n\n"
        f"Exact problem (do not change strategy intent, only fix the defect):\n{problem}"
    )


def _extract_python(text: str) -> str:
    """Pull the python code block out of a model response; fall back to the
    raw text when no fences are present."""
    if text is None:
        return ""
    blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", text, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip()


def write_strategy_code(track_dir: Path, strategy_id: str, code: str) -> Path:
    track_dir.mkdir(parents=True, exist_ok=True)
    path = track_dir / f"{strategy_id}.py"
    path.write_text(code)
    return path
