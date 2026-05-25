"""
cli.py
------
Operational CLI for launching and monitoring red-team engagements.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from typing import Any, Iterable, Optional

from langgraph.types import Command

from config import configure_tracing
from graph import build_graph
from memory import get_checkpointer, load_checkpoint_state, normalize_scope_entries
from state import initialize_state

LOGGER = logging.getLogger("redteam.cli")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _parse_targets(values: Iterable[str]) -> list[str]:
    targets: list[str] = []
    for entry in values:
        if not entry:
            continue
        targets.append(entry)
    return targets


def _validate_scope_match(provided: list[str], checkpoint_scope: list[str]) -> None:
    provided_normalized = normalize_scope_entries(provided)
    checkpoint_normalized = normalize_scope_entries(checkpoint_scope)
    if provided_normalized != checkpoint_normalized:
        raise RuntimeError(
            "SECURITY: Provided targets do not match checkpoint scope. "
            "Resume blocked to prevent scope drift."
        )


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


def _stream_graph(
    graph, input_payload, config, thread_id: str
) -> Optional[dict[str, Any]]:
    interrupt_payload: Optional[dict[str, Any]] = None
    for event in graph.stream(input_payload, config=config, stream_mode="debug"):
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        payload = event.get("payload", {})
        if event_type == "task":
            name = payload.get("name", "unknown")
            LOGGER.info("event=agent_active agent=%s thread_id=%s", name, thread_id)
            print(f"\n[agent] {name} active")
        elif event_type == "task_result":
            updates = _extract_updates(payload.get("result"))
            if payload.get("name") == "recon" and "recon_results" in updates:
                assets = updates["recon_results"].get("assets", [])
                LOGGER.info(
                    "event=recon_results thread_id=%s assets=%s",
                    thread_id,
                    len(assets),
                )
                _print_recon_results(updates["recon_results"])
            interrupts = payload.get("interrupts") or []
            if interrupts:
                first_interrupt = interrupts[0]
                if isinstance(first_interrupt, dict):
                    LOGGER.warning(
                        "event=hitl_interrupt thread_id=%s",
                        thread_id,
                    )
                    interrupt_payload = first_interrupt.get("value")
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


def _has_valid_checkpoint(snapshot: Any) -> bool:
    """
    Determine whether a state snapshot contains resumable data.
    """

    return any(
        [
            getattr(snapshot, "values", None),
            getattr(snapshot, "interrupts", None),
            getattr(snapshot, "next", None),
            getattr(snapshot, "tasks", None),
        ]
    )


def _resume_prompt(
    graph, config: dict[str, Any], thread_id: str
) -> Optional[dict[str, Any]]:
    snapshot = graph.get_state(config)
    if not _has_valid_checkpoint(snapshot):
        print(f"[HITL] No checkpoint found for thread ID {thread_id}.")
        return None
    if snapshot.values:
        pending = snapshot.values.get("exploitation_results", {}).get("pending_payload")
        if pending:
            _display_interrupt(
                {"message": "Review pending payload before resuming.", "pending_payload": pending},
                thread_id,
            )
    response = _prompt_yes_no("Approve exploitation payload execution now?")
    if response == "yes":
        return {"resume": "approved"}
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
        required=False,
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

    _configure_logging()

    if args.resume and not args.thread_id:
        parser.error("--resume requires --thread-id.")

    if args.checkpoint_backend:
        os.environ["PENTEST_CHECKPOINT_BACKEND"] = args.checkpoint_backend
    if args.checkpoint_path:
        os.environ["PENTEST_CHECKPOINT_PATH"] = args.checkpoint_path

    tracing_config = configure_tracing()
    if not tracing_config.enabled:
        missing = ",".join(tracing_config.missing) or "none"
        LOGGER.warning(
            "event=observability_disabled status=%s requested=%s missing=%s",
            tracing_config.status,
            tracing_config.requested,
            missing,
        )
    else:
        LOGGER.info(
            "event=observability_enabled project=%s endpoint=%s",
            tracing_config.project,
            tracing_config.endpoint,
        )

    checkpointer = get_checkpointer()
    thread_id = args.thread_id or uuid.uuid4().hex
    print(f"[session] Thread ID: {thread_id}")

    targets: list[str]
    if args.resume:
        checkpoint_state = load_checkpoint_state(checkpointer, thread_id)
        if not checkpoint_state:
            raise RuntimeError(
                f"No checkpoint data found for thread ID {thread_id}."
            )
        checkpoint_scope = checkpoint_state.get("scope") or []
        if not checkpoint_scope:
            raise RuntimeError(
                f"Checkpoint for thread ID {thread_id} has no stored scope."
            )

        if args.target:
            provided_targets = _parse_targets(args.target)
            _validate_scope_match(provided_targets, list(checkpoint_scope))
        targets = list(checkpoint_scope)
        LOGGER.info(
            "event=resume_scope_loaded thread_id=%s targets=%s",
            thread_id,
            targets,
        )
    else:
        if not args.target:
            parser.error("--target is required when starting a new engagement.")
        targets = _parse_targets(args.target)
        LOGGER.info("event=new_session thread_id=%s targets=%s", thread_id, targets)

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
        LOGGER.info("event=hitl_resume thread_id=%s", thread_id)
    else:
        input_payload = initialize_state(targets)

    while True:
        interrupt_payload = _stream_graph(graph, input_payload, run_config, thread_id)
        if not interrupt_payload:
            LOGGER.info("event=session_complete thread_id=%s", thread_id)
            print("\n[session] Engagement completed.")
            return 0
        _display_interrupt(interrupt_payload, thread_id)
        response = _prompt_yes_no("Approve exploitation payload execution now?")
        if response == "yes":
            LOGGER.info("event=hitl_resume thread_id=%s", thread_id)
            input_payload = Command(resume="approved")
            continue
        print(
            f"[HITL] Execution paused. Resume later with --resume --thread-id {thread_id}."
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
