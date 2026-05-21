"""
memory.py
---------
Persistent checkpointing utilities for LangGraph workflows.
"""

from __future__ import annotations

import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import ChannelVersions, Checkpoint, CheckpointMetadata
from langgraph.checkpoint.memory import InMemorySaver


DEFAULT_CHECKPOINT_PATH = Path(
    os.getenv("PENTEST_CHECKPOINT_PATH", ".data/pentest_checkpoints.pkl")
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
        backend = os.getenv("PENTEST_CHECKPOINT_BACKEND", "disk").strip().lower()
        path = Path(os.getenv("PENTEST_CHECKPOINT_PATH", str(DEFAULT_CHECKPOINT_PATH)))
        return cls(backend=backend, path=path)


class DiskBackedMemorySaver(InMemorySaver):
    """
    Disk-backed extension of the in-memory checkpointer for long-running workflows.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__()
        self._load_from_disk()

    def _snapshot(self) -> Dict[str, Any]:
        return {
            "storage": {
                thread_id: {
                    namespace: dict(checkpoints)
                    for namespace, checkpoints in namespaces.items()
                }
                for thread_id, namespaces in self.storage.items()
            },
            "writes": {key: dict(writes) for key, writes in self.writes.items()},
            "blobs": dict(self.blobs),
        }

    def _restore(self, payload: Dict[str, Any]) -> None:
        self.storage = defaultdict(lambda: defaultdict(dict))
        for thread_id, namespaces in payload.get("storage", {}).items():
            for namespace, checkpoints in namespaces.items():
                self.storage[thread_id][namespace].update(checkpoints)

        self.writes = defaultdict(dict)
        for key, writes in payload.get("writes", {}).items():
            self.writes[key].update(writes)

        self.blobs = dict(payload.get("blobs", {}))

    def _persist_to_disk(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_name(f"{self.path.name}.tmp")
        with temp_path.open("wb") as handle:
            pickle.dump(self._snapshot(), handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_path, self.path)

    def _load_from_disk(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("rb") as handle:
                payload = pickle.load(handle)
        except (OSError, pickle.UnpicklingError) as exc:  # pragma: no cover - safety net
            raise RuntimeError(
                f"Failed to load checkpoints from {self.path}."
            ) from exc
        self._restore(payload)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        updated = super().put(config, checkpoint, metadata, new_versions)
        self._persist_to_disk()
        return updated

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        super().put_writes(config, writes, task_id, task_path=task_path)
        self._persist_to_disk()

    def delete_thread(self, thread_id: str) -> None:
        super().delete_thread(thread_id)
        self._persist_to_disk()


def get_checkpointer(config: CheckpointConfig | None = None) -> InMemorySaver:
    """
    Build a checkpointer based on environment configuration.
    """

    config = config or CheckpointConfig.from_env()
    if config.backend == "memory":
        return InMemorySaver()
    return DiskBackedMemorySaver(config.path)
