"""
memory.py
---------
Persistent checkpointing utilities for LangGraph workflows.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from langgraph.checkpoint.memory import InMemorySaver
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:  # pragma: no cover - optional dependency
    SqliteSaver = None

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langgraph.checkpoint.sqlite import SqliteSaver


DEFAULT_CHECKPOINT_PATH = Path(".data") / "pentest_checkpoints.sqlite"


@dataclass(frozen=True)
class CheckpointConfig:
    """
    Configuration for checkpoint persistence.
    """

    backend: str
    path: Path

    @classmethod
    def from_env(cls) -> "CheckpointConfig":
        backend = os.getenv("PENTEST_CHECKPOINT_BACKEND", "disk").strip().lower()
        raw_path = os.getenv("PENTEST_CHECKPOINT_PATH")
        path = Path(raw_path) if raw_path else DEFAULT_CHECKPOINT_PATH
        path = path.expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        return cls(backend=backend, path=path)

def _build_sqlite_uri(path: Path) -> str:
    return f"sqlite:///{path.as_posix()}"


def get_checkpointer(
    config: CheckpointConfig | None = None,
) -> InMemorySaver | "SqliteSaver":
    """
    Build a checkpointer based on environment configuration.

    Note: the "disk" backend is an alias for the SQLite checkpointer.
    """

    config = config or CheckpointConfig.from_env()
    if config.backend == "memory":
        return InMemorySaver()
    if config.backend in {"sqlite", "disk"}:
        if SqliteSaver is None:  # pragma: no cover - safety net
            raise ImportError(
                "SqliteSaver is required for persistent checkpoints. "
                "Install LangGraph with sqlite support (e.g., "
                "pip install 'langgraph[sqlite]')."
            )
        config.path.parent.mkdir(parents=True, exist_ok=True)
        return SqliteSaver.from_conn_string(_build_sqlite_uri(config.path))
    raise ValueError(f"Unsupported checkpoint backend: {config.backend}")
