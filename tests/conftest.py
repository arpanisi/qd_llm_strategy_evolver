"""Test path setup: make the repo root importable and the data dir available.

Mirrors the sys.path bootstrap used by the scripts so tests run identically
from the repo root.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
