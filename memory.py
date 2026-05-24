"""
memory.py
---------
Persistent checkpointing utilities for LangGraph workflows.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:  # pragma: no cover - optional dependency
    SqliteSaver = None


DEFAULT_CHECKPOINT_PATH = Path(
    os.getenv("PENTEST_CHECKPOINT_PATH", ".data/pentest_checkpoints.sqlite")
)


@dataclass(frozen=True)
class CheckpointConfig:
    """
    Configuration for checkpoint persistence.
    """

    backend: str
    path: Path

    @classmethod
    def from_env(cls) -> "CheckpointConfig":
        backend = os.getenv("PENTEST_CHECKPOINT_BACKEND", "sqlite").strip().lower()
        path = Path(os.getenv("PENTEST_CHECKPOINT_PATH", str(DEFAULT_CHECKPOINT_PATH)))
        return cls(backend=backend, path=path)

def _sqlite_conn_string(path: Path) -> str:
    return f"sqlite:///{path.as_posix()}"


def get_checkpointer(config: CheckpointConfig | None = None):
    """
    Build a checkpointer based on environment configuration.
    """

    config = config or CheckpointConfig.from_env()
    if config.backend in {"memory", "in-memory", "mem"}:
        return InMemorySaver()
    if config.backend in {"sqlite", "disk"}:
        if SqliteSaver is None:  # pragma: no cover - safety net
            raise ImportError(
                "SqliteSaver is required for persistent checkpoints. "
                "Install langgraph with sqlite support."
            )
        config.path.parent.mkdir(parents=True, exist_ok=True)
        return SqliteSaver.from_conn_string(_sqlite_conn_string(config.path))
    raise ValueError(f"Unsupported checkpoint backend: {config.backend}")
