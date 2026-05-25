from __future__ import annotations

from typing import Any, Dict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

import agents
import graph as graph_module
import memory


class DummyLLM:
    def invoke(self, messages):  # noqa: D401 - minimal stub
        class Response:
            content = "{}"

        return Response()


class DummyTools:
    pass


def _fake_context(scope, model=None, base_url=None):  # noqa: ANN001
    return agents.AgentContext(llm=DummyLLM(), tools=DummyTools())


def _stub_vuln_agent(context: agents.AgentContext):
    def _node(state: Dict[str, Any]):
        scan_results = dict(state.get("scan_results", {}))
        scan_results["vuln_ran"] = True
        return {
            "scan_results": scan_results,
            "vulnerabilities": [
                {
                    "cve_id": "CVE-2024-0001",
                    "severity": "HIGH",
                    "cvss_score": 8.8,
                    "description": "Test CVE",
                    "affected_asset": "10.0.0.1:80",
                    "references": [],
                }
            ],
            "exploitation_results": {"exploitation_done": True},
        }

    return _node


def _stub_report_agent(context: agents.AgentContext):
    def _node(state: Dict[str, Any]):
        return {"report": "ok"}

    return _node


@pytest.mark.asyncio
async def test_orchestrator_routes_to_vuln(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(graph_module, "_build_context", _fake_context)
    monkeypatch.setattr(graph_module, "make_vuln_exploit_agent", _stub_vuln_agent)
    monkeypatch.setattr(graph_module, "make_reporting_agent", _stub_report_agent)

    graph = graph_module.build_graph(scope=["10.0.0.1"], checkpointer=InMemorySaver())
    state = {
        "scope": ["10.0.0.1"],
        "recon_results": {"assets": []},
        "scan_results": {"engagement_package": {}},
        "vulnerabilities": [],
        "exploitation_results": {},
        "report": "",
    }

    result = await graph.ainvoke(state, config={"configurable": {"thread_id": "test-thread"}})
    assert result["scan_results"]["vuln_ran"] is True


@pytest.mark.asyncio
async def test_hitl_interrupt_halts_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(graph_module, "_build_context", _fake_context)

    graph = graph_module.build_graph(scope=["10.0.0.1"], checkpointer=InMemorySaver())
    state = {
        "scope": ["10.0.0.1"],
        "recon_results": {"assets": []},
        "scan_results": {"engagement_package": {}},
        "vulnerabilities": [
            {
                "cve_id": "CVE-2024-0001",
                "severity": "HIGH",
                "cvss_score": 8.8,
                "description": "Test CVE",
                "affected_asset": "10.0.0.1:80",
                "references": [],
            }
        ],
        "exploitation_results": {"pending_payload": {"target_count": 1, "top_cves": ["CVE-2024-0001"]}},
        "report": "",
    }

    result = await graph.ainvoke(state, config={"configurable": {"thread_id": "test-thread"}})
    assert "__interrupt__" in result
    assert "pending_payload" in str(result["__interrupt__"])


@pytest.mark.asyncio
async def test_disk_checkpoint_resume_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(graph_module, "_build_context", _fake_context)
    monkeypatch.setattr(graph_module, "make_vuln_exploit_agent", _stub_vuln_agent)
    monkeypatch.setattr(graph_module, "make_reporting_agent", _stub_report_agent)

    state = {
        "scope": ["10.0.0.1"],
        "recon_results": {"assets": []},
        "scan_results": {"engagement_package": {}},
        "vulnerabilities": [
            {
                "cve_id": "CVE-2024-0001",
                "severity": "HIGH",
                "cvss_score": 8.8,
                "description": "Test CVE",
                "affected_asset": "10.0.0.1:80",
                "references": [],
            }
        ],
        "exploitation_results": {
            "pending_payload": {"target_count": 1, "top_cves": ["CVE-2024-0001"]}
        },
        "report": "",
    }

    thread_id = "disk-thread"
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        graph = graph_module.build_graph(scope=["10.0.0.1"], checkpointer=checkpointer)
        result = await graph.ainvoke(
            state, config={"configurable": {"thread_id": thread_id}}
        )
        assert "__interrupt__" in result

        checkpoint_tuple = await checkpointer.aget_tuple(
            {"configurable": {"thread_id": thread_id}}
        )
        assert checkpoint_tuple is not None
        saved_state = memory._extract_checkpoint_values(checkpoint_tuple)
        assert saved_state is not None
        assert saved_state.get("exploitation_results", {}).get("pending_payload")

        resume_result = await graph.ainvoke(
            Command(resume="approved"),
            config={"configurable": {"thread_id": thread_id}},
        )
        assert resume_result["report"] == "ok"
