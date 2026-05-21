"""
cli.py
------
Operational CLI for launching and monitoring red-team engagements.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from typing import Any, Iterable, Optional

from langgraph.types import Command

from config import configure_tracing
from graph import build_graph
from memory import get_checkpointer
from state import initialize_state


def _parse_targets(values: Iterable[str]) -> list[str]:
    targets: list[str] = []
    for entry in values:
        if not entry:
            continue
        targets.append(entry)
    return targets


def _print_recon_results(recon_results: dict[str, Any]) -> None:
    assets = recon_results.get("assets", [])
    if not assets:
        print("[recon] No assets discovered yet.")
        return
    for asset in assets:
        host = asset.get("host", "unknown")
        port = asset.get("port", "?")
        service = asset.get("service", "unknown")
        version = asset.get("version", "")
        version_suffix = f" ({version})" if version else ""
        print(f"[recon] {host}:{port} {service}{version_suffix}")


def _extract_updates(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        if "__root__" in result:
            return result.get("__root__", {}) or {}
        return result
    return {}


def _stream_graph(graph, input_payload, config) -> Optional[dict[str, Any]]:
    interrupt_payload: Optional[dict[str, Any]] = None
    for event in graph.stream(input_payload, config=config, stream_mode="debug"):
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        payload = event.get("payload", {})
        if event_type == "task":
            name = payload.get("name", "unknown")
            print(f"\n[agent] {name} active")
        elif event_type == "task_result":
            updates = _extract_updates(payload.get("result"))
            if payload.get("name") == "recon" and "recon_results" in updates:
                _print_recon_results(updates["recon_results"])
            interrupts = payload.get("interrupts") or []
            if interrupts:
                interrupt_payload = interrupts[0].get("value")  # type: ignore[assignment]
    return interrupt_payload


def _prompt_yes_no(message: str) -> str:
    while True:
        response = input(f"{message} [y/n]: ").strip().lower()
        if response in {"y", "yes"}:
            return "yes"
        if response in {"n", "no"}:
            return "no"
        print("Please respond with 'y' or 'n'.")


def _display_interrupt(interrupt_payload: dict[str, Any], thread_id: str) -> None:
    print("\n[HITL] Human approval required.")
    print(f"[HITL] Thread ID: {thread_id}")
    if isinstance(interrupt_payload, dict):
        message = interrupt_payload.get("message")
        if message:
            print(f"[HITL] {message}")
        pending = interrupt_payload.get("pending_payload")
        if pending:
            print("[HITL] Pending payload details:")
            print(json.dumps(pending, indent=2))


def _resume_prompt(
    graph, config: dict[str, Any], thread_id: str
) -> Optional[dict[str, Any]]:
    snapshot = graph.get_state(config)
    if snapshot.values:
        pending = snapshot.values.get("exploitation_results", {}).get("pending_payload")
        if pending:
            _display_interrupt(
                {"message": "Review pending payload before resuming.", "pending_payload": pending},
                thread_id,
            )
    response = _prompt_yes_no("Approve exploitation payload execution now?")
    if response == "yes":
        return {"resume": response}
    print(
        f"[HITL] Execution remains paused. Resume later with --resume --thread-id {thread_id}."
    )
    return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch and monitor a LangGraph-powered red team engagement."
    )
    parser.add_argument(
        "--target",
        action="append",
        required=True,
        help="Target host or subnet (repeatable).",
    )
    parser.add_argument("--thread-id", help="Existing thread ID to resume.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a stored checkpoint.",
    )
    parser.add_argument("--model", help="Override local LLM model name.")
    parser.add_argument("--base-url", help="Override local LLM base URL.")
    parser.add_argument(
        "--checkpoint-backend",
        choices=["disk", "memory"],
        help="Override checkpoint backend selection.",
    )
    parser.add_argument(
        "--checkpoint-path",
        help="Override checkpoint storage path.",
    )
    args = parser.parse_args(argv)

    if args.resume and not args.thread_id:
        parser.error("--resume requires --thread-id.")

    if args.checkpoint_backend:
        os.environ["PENTEST_CHECKPOINT_BACKEND"] = args.checkpoint_backend
    if args.checkpoint_path:
        os.environ["PENTEST_CHECKPOINT_PATH"] = args.checkpoint_path

    configure_tracing()

    targets = _parse_targets(args.target)
    thread_id = args.thread_id or uuid.uuid4().hex
    print(f"[session] Thread ID: {thread_id}")

    checkpointer = get_checkpointer()
    graph = build_graph(
        scope=targets,
        model=args.model,
        base_url=args.base_url,
        checkpointer=checkpointer,
    )
    run_config = {
        "configurable": {"thread_id": thread_id},
        "tags": ["pentest", "cli"],
        "metadata": {"scope": targets},
    }

    if args.resume:
        resume_payload = _resume_prompt(graph, run_config, thread_id)
        if not resume_payload:
            return 0
        input_payload = Command(resume=resume_payload["resume"])
    else:
        input_payload = initialize_state(targets)

    while True:
        interrupt_payload = _stream_graph(graph, input_payload, run_config)
        if not interrupt_payload:
            print("\n[session] Engagement completed.")
            return 0
        _display_interrupt(interrupt_payload, thread_id)
        response = _prompt_yes_no("Approve exploitation payload execution now?")
        if response == "yes":
            input_payload = Command(resume=response)
            continue
        print(
            f"[HITL] Execution paused. Resume later with --resume --thread-id {thread_id}."
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
