"""
config.py
---------
Environment configuration helpers for tracing and observability.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ENV_PATH = Path(os.getenv("PENTEST_ENV_PATH", ".env"))


@dataclass(frozen=True)
class TracingConfig:
    """
    LangSmith tracing configuration derived from environment variables.
    """

    enabled: bool
    endpoint: str
    project: str
    api_key: str
    hide_inputs: bool
    hide_outputs: bool

    @classmethod
    def from_env(cls) -> "TracingConfig":
        api_key = os.getenv("LANGCHAIN_API_KEY", "").strip()
        endpoint = os.getenv(
            "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
        ).strip()
        project = os.getenv("LANGCHAIN_PROJECT", "multi-agent-red-team").strip()
        tracing_flag = os.getenv("LANGCHAIN_TRACING_V2", "").strip().lower()
        enabled = tracing_flag in {"1", "true", "yes"} if tracing_flag else bool(api_key)
        hide_inputs = os.getenv("LANGCHAIN_HIDE_INPUTS", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        hide_outputs = os.getenv("LANGCHAIN_HIDE_OUTPUTS", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        return cls(
            enabled=enabled,
            endpoint=endpoint,
            project=project,
            api_key=api_key,
            hide_inputs=hide_inputs,
            hide_outputs=hide_outputs,
        )


def load_env(path: Path = DEFAULT_ENV_PATH) -> None:
    """
    Load environment variables from a .env file if present.
    """

    if not path.exists():
        return
    for line in path.read_text().splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        key, separator, value = entry.partition("=")
        if not key or not separator:
            continue
        cleaned = value.strip().strip("'").strip('"')
        os.environ.setdefault(key.strip(), cleaned)


def configure_tracing(env_path: Path | None = None) -> TracingConfig:
    """
    Initialize LangSmith tracing based on environment variables.
    """

    load_env(env_path or DEFAULT_ENV_PATH)
    config = TracingConfig.from_env()
    if config.enabled:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_ENDPOINT", config.endpoint)
        os.environ.setdefault("LANGCHAIN_PROJECT", config.project)
        if config.api_key:
            os.environ.setdefault("LANGCHAIN_API_KEY", config.api_key)
        if config.hide_inputs:
            os.environ.setdefault("LANGCHAIN_HIDE_INPUTS", "true")
        if config.hide_outputs:
            os.environ.setdefault("LANGCHAIN_HIDE_OUTPUTS", "true")
    return config
