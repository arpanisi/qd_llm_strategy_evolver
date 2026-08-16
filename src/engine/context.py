"""Shared simulation context dictionary for single-threaded Zipline execution."""

from __future__ import annotations

# Single-threaded simulation context shared across order wrappers, risk enforcement,
# and runtime execution handlers.
_CTX: dict = {}
