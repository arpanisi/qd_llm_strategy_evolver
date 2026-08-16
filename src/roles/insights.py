"""Island Insight Log and K-curation (Step 5/7 shared artifact).

Each candidate that reaches Step 7 contributes the evaluation's ``insight``.
Curation keeps the log focused: near-duplicate insights are merged (highest
weighted score wins) and only the top ``k`` distinct insights are retained per
island. The log feeds the Step 5 prompts of the next generation.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from src.roles.schemas import _overlap_ratio

_MAX_DUPLICATE_OVERLAP = 0.7


class InsightLog:
    def __init__(self, path: Path, k: int) -> None:
        self.path = path
        self.k = k
        self.entries: list[dict] = []   # {insight, score, strategy_id, generation}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                self.entries.append({
                    "insight": row["insight"],
                    "score": float(row["score"]),
                    "strategy_id": row["strategy_id"],
                    "generation": int(row["generation"]),
                })

    def add(self, insight: str, score: float, strategy_id: str, generation: int) -> None:
        insight = insight.strip()
        if not insight:
            return
        # merge with the most similar existing entry
        for entry in self.entries:
            if _overlap_ratio(entry["insight"], insight) > _MAX_DUPLICATE_OVERLAP:
                if score > entry["score"]:
                    entry.update(score=score, strategy_id=strategy_id,
                                 generation=generation)
                self._persist()
                return
        self.entries.append({
            "insight": insight,
            "score": score,
            "strategy_id": strategy_id,
            "generation": generation,
        })
        self._curate()

    def _curate(self) -> None:
        self.entries.sort(key=lambda e: e["score"], reverse=True)
        if len(self.entries) > self.k:
            self.entries = self.entries[: self.k]
        self._persist()

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["insight", "score", "strategy_id", "generation"]
            )
            writer.writeheader()
            writer.writerows(self.entries)

    def curate(self) -> None:
        """Step 7 consolidation pass (every K generations): merge near-duplicate
        insights, keep the top K by weighted score, persist."""
        merged: list[dict] = []
        for entry in sorted(self.entries, key=lambda e: e["score"], reverse=True):
            hit = None
            for existing in merged:
                if _overlap_ratio(existing["insight"], entry["insight"]) > _MAX_DUPLICATE_OVERLAP:
                    hit = existing
                    break
            if hit is None:
                merged.append(dict(entry))
            elif entry["score"] > hit["score"]:
                hit["score"] = entry["score"]
                hit["strategy_id"] = entry["strategy_id"]
                hit["generation"] = entry["generation"]
        self.entries = merged[: self.k]
        self._persist()

    def recent(self, limit: int = 25) -> list[str]:
        return [e["insight"] for e in self.entries[:limit]]

    def snapshot(self) -> Optional[dict]:
        """Best entry overall, used as the run's final insight summary."""
        if not self.entries:
            return None
        top = self.entries[0]
        return {"insight": top["insight"], "score": top["score"]}
