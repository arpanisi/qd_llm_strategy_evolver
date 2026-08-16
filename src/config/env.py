"""Environment loading for the evolver.

Reads KEY=VALUE lines from ``.env`` into ``os.environ`` if not already set.
Credential names are locked by the coding plan: OPENROUTER_API_KEY,
DEEPSEEK_API_KEY, WRDS_USERID, WRDS_PGPASS.
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_env_file() -> Path | None:
    root = Path(__file__).resolve().parents[2]
    env = root / ".env"
    return env if env.exists() else None


def load_env() -> None:
    env_path = _find_env_file()
    if env_path is None:
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def normalize_wrds_password(value: str | None) -> str:
    """Return a raw password or a pgpass-style entry's password field."""
    if value is None:
        return ""
    stripped = value.strip()
    parts = stripped.split(":", 4)
    if len(parts) == 5 and parts[0] and parts[1] and parts[2] and parts[3]:
        return parts[4]
    return stripped


def require_env(key: str) -> str:
    load_env()
    value = os.environ.get(key, "")
    if not value:
        raise RuntimeError(f"Required environment variable {key!r} is missing.")
    return value


def wrds_credentials() -> tuple[str, str]:
    load_env()
    user = os.environ.get("WRDS_USERID", "")
    password = normalize_wrds_password(os.environ.get("WRDS_PGPASS"))
    return user, password
