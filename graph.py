"""
graph.py
--------
LangGraph topology and routing logic for the autonomous multi-agent red team.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from agents import (
    AgentContext,
    LocalLLM,
    make_orchestrator,
    make_post_exploitation_agent,
    make_recon_agent,
    make_reporting_agent,
    make_vuln_exploit_agent,
)
from memory import get_checkpointer
from state import PentestState
from tools import DockerSandbox, SandboxConfig, ScopedToolset

LOGGER = logging.getLogger("redteam.graph")

APPROVED_RESPONSES = {"y", "yes", "approve", "approved", "true"}


def _build_context(
    scope: list[str],
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> AgentContext:
    """
    Construct the shared agent context with a local LLM and scoped toolset.
    """

    llm = LocalLLM(model=model or os.getenv("LOCAL_LLM_MODEL", "llama3"), base_url=base_url)
    sandbox = DockerSandbox(scope, SandboxConfig())
    tools = ScopedToolset(sandbox)
    return AgentContext(llm=llm, tools=tools)


def _hitl_approval_node(state: PentestState) -> Dict[str, Any]:
    """
    Pause execution for human approval before active exploitation.
    """

    pending = state.get("exploitation_results", {}).get("pending_payload")
    if not pending:
        return {}

    LOGGER.warning("event=hitl_interrupt")

    response = interrupt(
        {
            "type": "approval",
            "message": "Approve exploitation payload execution? (yes/no)",
            "pending_payload": pending,
        }
    )
    approved = str(response).strip().lower() in APPROVED_RESPONSES
    exploitation_results = dict(state.get("exploitation_results", {}))
    exploitation_results["approved"] = approved
    LOGGER.info("event=hitl_approval approved=%s", approved)
    return {"exploitation_results": exploitation_results}


def build_graph(
    scope: list[str],
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    checkpointer=None,
):
    """
    Build and compile the LangGraph StateGraph for the engagement.
    """

    context = _build_context(scope, model=model, base_url=base_url)
    graph = StateGraph(PentestState)

    orchestrator_node = make_orchestrator(context)

    def _logged_orchestrator(state: PentestState) -> Dict[str, Any]:
        updates = orchestrator_node(state)
        decision = updates.get("scan_results", {}).get("orchestrator_decision")
        if decision:
            LOGGER.info("event=route_decision decision=%s", decision)
        return updates

    graph.add_node("orchestrator", _logged_orchestrator)
    graph.add_node("recon", make_recon_agent(context))
    graph.add_node("vuln_exploit", make_vuln_exploit_agent(context))
    graph.add_node("hitl_approval", _hitl_approval_node)
    graph.add_node("post_exploitation", make_post_exploitation_agent(context))
    graph.add_node("report", make_reporting_agent(context))

    graph.set_entry_point("orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        lambda state: state.get("scan_results", {}).get("orchestrator_decision", "report"),
        {
            "recon": "recon",
            "vuln": "vuln_exploit",
            "hitl": "hitl_approval",
            "post": "post_exploitation",
            "report": "report",
        },
    )

    graph.add_edge("recon", "orchestrator")
    graph.add_edge("vuln_exploit", "orchestrator")
    graph.add_edge("post_exploitation", "orchestrator")

    graph.add_conditional_edges(
        "hitl_approval",
        lambda state: "vuln_exploit"
        if state.get("exploitation_results", {}).get("approved")
        else "report",
        {
            "vuln_exploit": "vuln_exploit",
            "report": "report",
        },
    )

    graph.add_edge("report", END)

    resolved_checkpointer = checkpointer or get_checkpointer()
    return graph.compile(checkpointer=resolved_checkpointer)
