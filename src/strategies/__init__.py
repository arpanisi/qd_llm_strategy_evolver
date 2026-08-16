"""Strategy corpus: the 8 baselines (4 per track) and seed strategies live here.

Each strategy is a self-contained source file exposing `initialize` plus
`handle_data`/`before_trading_start`/scheduled callbacks, executed by
src.engine.runtime.run_strategy with the standard per-track cost models and
constraint-enforcing order wrappers.
"""
