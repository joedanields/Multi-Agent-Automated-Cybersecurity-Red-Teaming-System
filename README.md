# Multi-Agent Automated Cybersecurity Red Teaming System

Autonomous LangGraph-driven red team orchestration with isolated Docker execution, strict scope enforcement, and human-in-the-loop approvals.

## Architecture Overview

- **LangGraph topology**: Orchestrator routes to Recon, Vulnerability/Exploit, HITL Approval, Post-Exploitation, and Reporting nodes.
- **State model**: `PentestState` is the single source of truth for routing and checkpoint persistence.
- **Isolation**: The Kali sandbox runs only on `sandbox-net` while the host orchestrator operates on `decepticon-net`. All offensive tooling is executed inside the container.
- **Scope enforcement**: Targets are validated against explicit IP/CIDR/hostname scope checks. Out-of-scope attempts are blocked.

## Setup Instructions

### Prerequisites

- Python 3.11+
- Docker Desktop (or Docker Engine)

### Environment Variables

Copy `.env.example` to `.env` and set values:

- `LANGCHAIN_TRACING_V2=true`
- `LANGCHAIN_API_KEY=...` (required for LangSmith)
- `LANGCHAIN_PROJECT=multi-agent-red-team`
- `LANGCHAIN_ENDPOINT=https://api.smith.langchain.com`
- `LANGCHAIN_HIDE_INPUTS` / `LANGCHAIN_HIDE_OUTPUTS` (optional)
- `PENTEST_CHECKPOINT_BACKEND=disk` (or `memory`)
- `PENTEST_CHECKPOINT_PATH=.data/pentest_checkpoints.pkl`

### Virtual Environment

```
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .[test]
```

## Execution Commands

### Start a New Engagement

```
python cli.py --target 192.168.1.0/24
```

You can pass multiple targets:

```
python cli.py --target 10.0.0.1 --target 10.0.0.2
```

The CLI prints a `thread_id` for resumable HITL checkpoints.

### Resume After HITL Pause

```
python cli.py --resume --thread-id <thread_id>
```

If you provide `--target` with `--resume`, it must match the checkpoint scope exactly. Mismatches are blocked to prevent scope drift:

```
python cli.py --resume --thread-id <thread_id> --target 192.168.1.0/24
```

When prompted, approve the payload to continue. The system resumes with `Command(resume="approved")`.

## Observability

LangSmith tracing is enabled only when both `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` are present. If disabled or misconfigured, the CLI emits a high-visibility warning.

## CI/CD and Testing

Run checks locally:

```
ruff check .
bandit -r . -x .venv
pytest -q
```

The GitHub Actions workflow executes the same steps on each push and pull request to `main`.
