"""
memory.py
---------
Persistent checkpointing utilities for LangGraph workflows.
"""

from __future__ import annotations

import ipaddress
import os
import pickle
import socket
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

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
                f"Failed to load checkpoints from {self.path}: "
                f"{type(exc).__name__}: {exc}"
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


def _extract_checkpoint_values(checkpoint: Any) -> Optional[Dict[str, Any]]:
    if checkpoint is None:
        return None
    if hasattr(checkpoint, "checkpoint"):
        checkpoint = checkpoint.checkpoint
    if isinstance(checkpoint, tuple) and checkpoint:
        checkpoint = checkpoint[0]
    if isinstance(checkpoint, dict):
        if "values" in checkpoint and isinstance(checkpoint["values"], dict):
            return checkpoint["values"]
        if "channel_values" in checkpoint and isinstance(
            checkpoint["channel_values"], dict
        ):
            channel_values = checkpoint["channel_values"]
            if "__root__" in channel_values and isinstance(
                channel_values["__root__"], dict
            ):
                return channel_values["__root__"]
            return channel_values
        if "state" in checkpoint and isinstance(checkpoint["state"], dict):
            return checkpoint["state"]
    return None


def load_checkpoint_state(
    checkpointer: InMemorySaver, thread_id: str
) -> Optional[Dict[str, Any]]:
    """
    Load the most recent checkpointed state for a thread.
    """

    if not thread_id:
        return None

    config = {"configurable": {"thread_id": thread_id}}
    if hasattr(checkpointer, "get_tuple"):
        try:
            checkpoint_tuple = checkpointer.get_tuple(config)
        except TypeError:
            checkpoint_tuple = None
        values = _extract_checkpoint_values(checkpoint_tuple)
        if values is not None:
            return values

    if hasattr(checkpointer, "get"):
        try:
            checkpoint = checkpointer.get(config)
        except TypeError:
            checkpoint = None
        values = _extract_checkpoint_values(checkpoint)
        if values is not None:
            return values

    storage = getattr(checkpointer, "storage", None)
    if storage and thread_id in storage:
        namespaces = storage.get(thread_id, {})
        for checkpoints in namespaces.values():
            if not checkpoints:
                continue
            last_checkpoint = next(reversed(checkpoints.values()))
            values = _extract_checkpoint_values(last_checkpoint)
            if values is not None:
                return values

    return None


def _resolve_hostname(
    hostname: str,
) -> Iterable[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        info = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise RuntimeError(
            f"Hostname {hostname} could not be resolved for scope validation."
        ) from exc

    addresses = set()
    for _, _, _, _, sockaddr in info:
        ip_text = sockaddr[0]
        if "%" in ip_text:
            ip_text = ip_text.split("%", 1)[0]
        try:
            addresses.add(ipaddress.ip_address(ip_text))
        except ValueError:
            continue
    if not addresses:
        raise RuntimeError(f"Hostname {hostname} resolved to no valid IP addresses.")
    return addresses


def normalize_scope_entries(scope: Sequence[str]) -> set[str]:
    """
    Normalize a scope list into canonical network strings for comparison.
    """

    normalized: set[str] = set()
    for entry in scope:
        entry = entry.strip()
        if not entry:
            continue
        if "/" in entry:
            network = ipaddress.ip_network(entry, strict=False)
            normalized.add(str(network))
            continue
        try:
            ip = ipaddress.ip_address(entry)
        except ValueError:
            for resolved in _resolve_hostname(entry):
                network = ipaddress.ip_network(
                    f"{resolved}/{resolved.max_prefixlen}", strict=False
                )
                normalized.add(str(network))
            continue
        normalized.add(
            str(ipaddress.ip_network(f"{ip}/{ip.max_prefixlen}", strict=False))
        )

    return normalized
