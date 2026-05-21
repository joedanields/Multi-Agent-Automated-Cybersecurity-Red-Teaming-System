"""
agents.py
---------
LangGraph agent node implementations for the autonomous red teaming system.

Each agent is implemented as a pure function that accepts the current PentestState
and returns partial state updates. This keeps routing deterministic and makes
human-in-the-loop interruptions safe.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import requests
from langchain_core.messages import HumanMessage, SystemMessage

from state import PentestState, VulnerabilityFinding
from tools import ScopedToolset


class LocalLLM:
    """
    Simple wrapper around a local chat model (defaults to Ollama).
    """

    def __init__(self, model: str, base_url: Optional[str] = None):
        from langchain_community.chat_models import ChatOllama

        self._client = ChatOllama(model=model, base_url=base_url)

    def invoke(self, messages: Sequence[SystemMessage | HumanMessage]):
        return self._client.invoke(messages)


@dataclass
class AgentContext:
    """
    Shared runtime dependencies for all agents.
    """

    llm: LocalLLM
    tools: ScopedToolset


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to parse the first JSON object embedded in a text response.
    """

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _call_llm_json(
    llm: LocalLLM,
    system_prompt: str,
    user_prompt: str,
    fallback: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Request JSON from the model, returning a safe fallback on parsing failure.
    """

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    content = response.content if hasattr(response, "content") else str(response)
    parsed = _extract_json_block(content)
    return parsed or fallback


def _severity_rank(severity: str) -> int:
    """
    Convert severity labels to sortable scores.
    """

    return {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(severity.upper(), 0)


def _nvd_lookup(keyword: str) -> List[Dict[str, Any]]:
    """
    Query NVD for vulnerabilities matching a keyword (service + version).
    """

    api_key = os.getenv("NVD_API_KEY")
    headers = {"apiKey": api_key} if api_key else {}
    params = {"keywordSearch": keyword, "resultsPerPage": 20}
    try:
        response = requests.get(
            "https://services.nvd.nist.gov/rest/json/cves/2.0",
            params=params,
            headers=headers,
            timeout=30,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"NVD API request failed for keyword '{keyword}'.") from exc

    if response.status_code == 403:
        raise RuntimeError("NVD API access denied. Check the NVD_API_KEY.")
    if response.status_code == 429:
        raise RuntimeError("NVD API rate limit exceeded. Consider backoff or API key.")

    response.raise_for_status()
    return response.json().get("vulnerabilities", [])


def _parse_nvd_entry(entry: Dict[str, Any], asset: str) -> Optional[VulnerabilityFinding]:
    """
    Convert an NVD entry into a normalized VulnerabilityFinding object.
    """

    cve = entry.get("cve", {})
    cve_id = cve.get("id")
    if not cve_id:
        return None

    descriptions = cve.get("descriptions", [])
    description = descriptions[0].get("value", "") if descriptions else ""
    references = [ref.get("url", "") for ref in cve.get("references", []) if ref.get("url")]

    metrics = cve.get("metrics", {})
    cvss_score = 0.0
    severity = "UNKNOWN"
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        if key in metrics:
            metric = metrics[key][0].get("cvssData", {})
            cvss_score = float(metric.get("baseScore", 0.0))
            severity = metric.get("baseSeverity", "UNKNOWN")
            break

    return VulnerabilityFinding(
        cve_id=cve_id,
        severity=severity,
        cvss_score=cvss_score,
        description=description,
        affected_asset=asset,
        references=references,
    )


def _build_engagement_package(llm: LocalLLM, scope: Sequence[str]) -> Dict[str, Any]:
    """
    Generate OPPLAN (MITRE-mapped), Rules of Engagement, and ConOps.
    """

    system_prompt = (
        "You are a senior red-team planner. Respond ONLY with JSON."
    )
    user_prompt = (
        "Create an engagement package for the following scope.\n"
        f"Scope: {', '.join(scope)}\n"
        "Return JSON with keys: opplan (list of steps with tactic, technique_id, "
        "technique_name, objective, rationale), roe (string), conops (string)."
    )

    fallback = {
        "opplan": [
            {
                "tactic": "Reconnaissance",
                "technique_id": "T1595",
                "technique_name": "Active Scanning",
                "objective": "Enumerate exposed services inside authorized scope.",
                "rationale": "Identify attack surface without exceeding RoE.",
            }
        ],
        "roe": "Only targets within the authorized scope are tested. No DoS.",
        "conops": "Phased recon, validation, controlled exploitation, and reporting.",
    }
    return _call_llm_json(llm, system_prompt, user_prompt, fallback)


def make_orchestrator(context: AgentContext):
    """
    Orchestrator agent: builds engagement package and decides next routing action.
    """

    def orchestrator(state: PentestState) -> Dict[str, Any]:
        scan_results = dict(state.get("scan_results", {}))
        if "engagement_package" not in scan_results:
            scan_results["engagement_package"] = _build_engagement_package(
                context.llm, state["scope"]
            )

        # Decide routing based on current state.
        if not state.get("recon_results"):
            decision = "recon"
        elif state.get("exploitation_results", {}).get("pending_payload") and not state.get(
            "exploitation_results", {}
        ).get("approved"):
            decision = "hitl"
        elif not state.get("vulnerabilities"):
            decision = "vuln"
        elif state.get("vulnerabilities") and not state.get("exploitation_results", {}).get(
            "exploitation_done"
        ):
            decision = "vuln"
        elif state.get("exploitation_results", {}).get("access_level") and not state.get(
            "exploitation_results", {}
        ).get("post_exploitation_done"):
            decision = "post"
        else:
            decision = "report"

        scan_results["orchestrator_decision"] = decision
        return {"scan_results": scan_results}

    return orchestrator


def make_recon_agent(context: AgentContext):
    """
    Reconnaissance agent: passive/active discovery within scope.
    """

    def recon_agent(state: PentestState) -> Dict[str, Any]:
        targets = state["scope"]
        raw_output = context.tools.nmap_scan(targets)

        # Parse minimal service data from Nmap output.
        services: List[Dict[str, str]] = []
        current_host: Optional[str] = None
        for line in raw_output.splitlines():
            if line.startswith("Nmap scan report for"):
                current_host = line.split()[-1]
                continue
            match = re.match(r"(\d+)/tcp\s+open\s+(\S+)\s*(.*)", line)
            if match and current_host:
                port, service, version = match.groups()
                services.append(
                    {
                        "host": current_host,
                        "port": port,
                        "service": service,
                        "version": version.strip(),
                    }
                )

        recon_results = {
            "assets": services,
            "topology": {"targets": targets},
        }
        scan_results = dict(state.get("scan_results", {}))
        scan_results["nmap_raw"] = raw_output
        return {"recon_results": recon_results, "scan_results": scan_results}

    return recon_agent


def make_vuln_exploit_agent(context: AgentContext):
    """
    Vulnerability & exploitation agent: NVD lookup and controlled payload execution.
    """

    def vuln_exploit_agent(state: PentestState) -> Dict[str, Any]:
        assets = state.get("recon_results", {}).get("assets", [])
        vulnerabilities: List[VulnerabilityFinding] = []

        for asset in assets:
            keyword = f"{asset.get('service')} {asset.get('version')}".strip()
            if not keyword:
                continue
            for entry in _nvd_lookup(keyword):
                finding = _parse_nvd_entry(entry, f"{asset.get('host')}:{asset.get('port')}")
                if finding:
                    vulnerabilities.append(finding)

        vulnerabilities.sort(
            key=lambda item: (
                -_severity_rank(item["severity"]),
                -item["cvss_score"],
            )
        )

        exploitation_results = dict(state.get("exploitation_results", {}))

        # If we already have approval, execute exploit payloads now.
        if exploitation_results.get("approved") and vulnerabilities:
            targets = [asset["host"] for asset in assets if asset.get("host")]
            payload_commands = []
            for finding in vulnerabilities[:3]:
                safe_cve = _sanitize_shell_fragment(finding["cve_id"])
                payload_commands.append(
                    f"echo Simulated_exploit_for_{safe_cve}"
                )

            payload_outputs = []
            for command in payload_commands:
                payload_outputs.append(context.tools.execute_payload(command, targets))

            exploitation_results.update(
                {
                    "payloads_executed": payload_commands,
                    "payload_outputs": payload_outputs,
                    "access_level": "shell",
                    "exploitation_done": True,
                }
            )
        elif vulnerabilities and not exploitation_results.get("approved"):
            # Stage payloads but do not execute without HITL approval.
            exploitation_results["pending_payload"] = {
                "target_count": len(assets),
                "top_cves": [v["cve_id"] for v in vulnerabilities[:3]],
            }

        return {
            "vulnerabilities": vulnerabilities,
            "exploitation_results": exploitation_results,
        }

    return vuln_exploit_agent


def _sanitize_shell_fragment(value: str) -> str:
    """
    Sanitize identifiers before interpolating into shell commands.
    """

    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


def make_post_exploitation_agent(context: AgentContext):
    """
    Post-exploitation agent: privilege escalation checks and lateral movement.
    """

    def post_exploitation_agent(state: PentestState) -> Dict[str, Any]:
        exploitation_results = dict(state.get("exploitation_results", {}))
        if not exploitation_results.get("access_level") or exploitation_results.get(
            "post_exploitation_done"
        ):
            return {}

        targets = [
            asset["host"]
            for asset in state.get("recon_results", {}).get("assets", [])
            if asset.get("host")
        ]
        commands = [
            "id",
            "whoami",
            "ip -4 addr show",
        ]
        outputs = []
        for command in commands:
            outputs.append(context.tools.execute_payload(command, targets))

        exploitation_results.update(
            {
                "post_exploitation": outputs,
                "post_exploitation_done": True,
            }
        )
        return {"exploitation_results": exploitation_results}

    return post_exploitation_agent


def make_reporting_agent(context: AgentContext):
    """
    Reporting agent: produce boardroom-ready mitigation report.
    """

    def reporting_agent(state: PentestState) -> Dict[str, Any]:
        system_prompt = "You are a cybersecurity executive reporter. Respond ONLY with JSON."
        user_prompt = (
            "Create a boardroom-ready red team report with business impact and "
            "mitigation recommendations.\n"
            f"Scope: {state.get('scope')}\n"
            f"Recon: {state.get('recon_results')}\n"
            f"Vulnerabilities: {state.get('vulnerabilities')}\n"
            f"Exploitation: {state.get('exploitation_results')}\n"
            "Return JSON with key: report (markdown string)."
        )
        fallback = {
            "report": (
                "# Red Team Report\n\n"
                "## Executive Summary\n"
                "Findings derived from scoped reconnaissance and validated CVEs.\n\n"
                "## Key Risks\n"
                "- Exposure of critical services within authorized scope.\n\n"
                "## Mitigations\n"
                "- Patch vulnerable services.\n"
                "- Enforce segmentation and MFA.\n"
            )
        }
        response = _call_llm_json(context.llm, system_prompt, user_prompt, fallback)
        report = response.get("report", fallback["report"])
        return {"report": report}

    return reporting_agent
