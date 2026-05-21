"""
state.py
---------
Centralized, persistent state definition for the LangGraph-driven red teaming system.

The StateGraph uses this TypedDict as the single source of truth shared across all
agents. Keep this file lightweight and stable to avoid breaking graph serialization
or state persistence in production.
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class VulnerabilityFinding(TypedDict):
    """
    Structured vulnerability record to keep CVE data consistent across agents.
    """

    cve_id: str
    severity: str
    cvss_score: float
    description: str
    affected_asset: str
    references: List[str]


class PentestState(TypedDict):
    """
    The LangGraph shared state for the autonomous red team.

    NOTE: This schema must remain stable because it is the only trusted record
    persisted between graph steps and human-in-the-loop interrupts.
    """

    # Authorized targets (IPs/subnets) for this engagement.
    scope: List[str]

    # Aggregated discovery: open ports, service versions, and network topology.
    recon_results: Dict[str, Any]

    # Raw outputs from active scanning tools (e.g., Nmap output, banner grabs).
    scan_results: Dict[str, Any]

    # Prioritized CVE list derived from NVD and other sources.
    vulnerabilities: List[VulnerabilityFinding]

    # Exploitation telemetry: payloads, response deltas, access levels.
    exploitation_results: Dict[str, Any]

    # Final boardroom-ready report in Markdown.
    report: str


def initialize_state(scope: List[str]) -> PentestState:
    """
    Create a clean starting state for a new engagement.
    """

    return PentestState(
        scope=scope,
        recon_results={},
        scan_results={},
        vulnerabilities=[],
        exploitation_results={},
        report="",
    )
