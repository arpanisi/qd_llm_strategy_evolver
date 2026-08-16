"""Tier runner: builds the per-(track, path) engine and executes a tier.

Exposed as a function so the thin run_tierN.py scripts stay declarative.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from src.config.settings import load_config
from src.evolution.orchestrator import tier_specs
from src.evolution.path_a_run import PathARunner
from src.evolution.path_b_run import PathBRunner

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs" / "evolution"


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )


def run_tier(tier: str, track_name: str | None = None, path: str | None = None) -> dict:
    cfg = load_config()
    _configure_logging(cfg.log_level)
    spec = tier_specs()[tier]
    results: dict = {}
    for track in cfg.tracks:
        if track_name and track.name != track_name:
            continue
        results[track.name] = {}
        for p in ("a", "b"):
            if path and p != path:
                continue
            runner = (
                PathARunner(cfg, track, p, spec)
                if p == "a" else PathBRunner(cfg, track, p, spec)
            )
            summary = runner.run()
            results[track.name][p] = summary
            with open(runner.run_dir / "run.json", "w") as fh:
                json.dump({"summary": summary, "completed": True}, fh, indent=2, sort_keys=True)
    return results


def tier_name(tier: str) -> str:
    return tier_specs()[tier].name
