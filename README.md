# Multi-Agent-Automated-Cybersecurity-Red-Teaming-System
an autonomous "Vibe Hacking" multi-agent ecosystem designed to proactively simulate coordinated cyberattacks and identify systemic vulnerabilities before malicious actors exploit them.

## Observability (LangSmith)
Tracing is configured via environment variables (see `.env.example` for a template):

- `LANGCHAIN_TRACING_V2` (set to `true` to enable tracing)
- `LANGCHAIN_ENDPOINT` (defaults to `https://api.smith.langchain.com`)
- `LANGCHAIN_API_KEY` (LangSmith API key)
- `LANGCHAIN_PROJECT` (project name for grouping runs)
- `LANGCHAIN_HIDE_INPUTS` / `LANGCHAIN_HIDE_OUTPUTS` (optional redaction toggles)

## Checkpointing
Checkpoint persistence can be configured via:

- `PENTEST_CHECKPOINT_BACKEND` (`disk` or `memory`)
- `PENTEST_CHECKPOINT_PATH` (file path for disk-backed checkpoints)

## CLI
Launch an engagement via:

```
python cli.py --target 192.168.1.0/24
```
