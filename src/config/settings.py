"""Typed run configuration loaded from ``config/evolver.yaml`` plus the .env.

Every locked parameter from coding-plan.md lives here so the run is fully
self-describing and reproducible.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.config.env import load_env

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "evolver.yaml"


def _date(value: str) -> dt.date:
    return dt.date.fromisoformat(str(value))


@dataclass(frozen=True)
class DateRange:
    start: dt.date
    end: dt.date

    @classmethod
    def from_list(cls, pair: list[str]) -> "DateRange":
        return cls(start=_date(pair[0]), end=_date(pair[1]))


@dataclass(frozen=True)
class EquitiesCost:
    per_share_cost: float
    min_trade_cost: float
    volume_limit: float
    price_impact: float
    short_margin: float
    gross_leverage_cap: float


@dataclass(frozen=True)
class FuturesCost:
    commission_per_contract: float
    slippage_ticks: int
    margin_fraction: float
    gross_leverage_cap: float


@dataclass(frozen=True)
class TrackConfig:
    name: str
    bundle_name: str
    calendar: str
    bars_per_year: int
    starting_cash: float
    train: DateRange
    validation: DateRange
    test: DateRange
    cost: EquitiesCost | FuturesCost
    # Equities-only
    n_names: int | None = None
    universe_date: dt.date | None = None
    data_end: dt.date | None = None
    # Futures-only
    instruments: tuple[str, ...] = ()
    yfinance_tickers: tuple[str, ...] = ()
    point_value: dict[str, float] = field(default_factory=dict)
    tick_size: float | None = None
    roll_flag_threshold: float | None = None

    @property
    def is_futures(self) -> bool:
        return self.name == "futures"

    @property
    def sqr_bars_per_year(self) -> float:
        return float(self.bars_per_year) ** 0.5


@dataclass(frozen=True)
class PathAConfig:
    alpha: float
    n_best_cousins: int
    n_diverse_cousins: int
    n_random_cousins: int
    cousin_neighbor_sigma: float
    bitflips: int
    resample_attempts: int
    bins_per_dim: int
    bin_headroom: float
    ir_near_tie: float
    turnover_tiebreak: str

    @property
    def n_cousins(self) -> int:
        return self.n_best_cousins + self.n_diverse_cousins + self.n_random_cousins


@dataclass(frozen=True)
class PathBConfig:
    pop_target: int
    das_dennis_p: int
    n_objectives: int
    nadir_floor: float
    distance_near_tie: float


@dataclass(frozen=True)
class EvolutionConfig:
    num_islands: int
    generations: int
    candidates_per_island_per_gen: int
    migration_interval: int
    curation_interval: int


@dataclass(frozen=True)
class ModelConfig:
    openrouter_base_url: str
    research_model: str
    implementation_model: str
    evaluation_model: str
    max_refinement_attempts: int
    request_timeout_s: int
    max_retries: int


@dataclass(frozen=True)
class RunConfig:
    seed: int
    log_level: str
    equities: TrackConfig
    futures: TrackConfig
    evolution: EvolutionConfig
    path_a: PathAConfig
    path_b: PathBConfig
    models: ModelConfig

    @property
    def tracks(self) -> list[TrackConfig]:
        return [self.equities, self.futures]

    def track(self, name: str) -> TrackConfig:
        for t in self.tracks:
            if t.name == name:
                return t
        raise KeyError(name)

    @classmethod
    def from_yaml(cls, path: Path | str = DEFAULT_CONFIG_PATH) -> "RunConfig":
        load_env()
        with open(path, "r") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle)
        run = raw["run"]
        tracks = raw["tracks"]
        evolution = raw["evolution"]
        path_a = raw["path_a"]
        path_b = raw["path_b"]
        models = raw["models"]

        eq = tracks["equities"]
        fx = tracks["futures"]
        equities = TrackConfig(
            name="equities",
            bundle_name=eq["bundle_name"],
            calendar=eq["calendar"],
            bars_per_year=eq["bars_per_year"],
            starting_cash=float(eq["starting_cash"]),
            n_names=eq["n_names"],
            universe_date=_date(eq["universe_date"]),
            data_end=_date(eq["data_end"]),
            train=DateRange.from_list(eq["train"]),
            validation=DateRange.from_list(eq["validation"]),
            test=DateRange.from_list(eq["test"]),
            cost=EquitiesCost(**eq["cost"]),
        )
        futures = TrackConfig(
            name="futures",
            bundle_name=fx["bundle_name"],
            calendar=fx["calendar"],
            bars_per_year=fx["bars_per_year"],
            starting_cash=float(fx["starting_cash"]),
            instruments=tuple(fx["instruments"]),
            yfinance_tickers=tuple(fx["yfinance_tickers"]),
            point_value={k: float(v) for k, v in fx["point_value"].items()},
            tick_size=float(fx["tick_size"]),
            roll_flag_threshold=float(fx["roll_flag_threshold"]),
            train=DateRange.from_list(fx["train"]),
            validation=DateRange.from_list(fx["validation"]),
            test=DateRange.from_list(fx["test"]),
            cost=FuturesCost(**fx["cost"]),
        )
        return cls(
            seed=int(run["seed"]),
            log_level=run["log_level"],
            equities=equities,
            futures=futures,
            evolution=EvolutionConfig(**evolution),
            path_a=PathAConfig(**path_a),
            path_b=PathBConfig(**path_b),
            models=ModelConfig(**models),
        )


def load_config() -> RunConfig:
    return RunConfig.from_yaml()
