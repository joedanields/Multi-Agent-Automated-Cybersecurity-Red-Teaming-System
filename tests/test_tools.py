from __future__ import annotations

import socket
from typing import Any, Dict, List, Optional

import pytest
from docker.errors import NotFound

import tools


class FakeExecResult:
    def __init__(self, exit_code: int = 0, output: str = "") -> None:
        self.exit_code = exit_code
        self.output = output.encode()


class FakeNetwork:
    def __init__(self, name: str) -> None:
        self.name = name

    def disconnect(self, container: "FakeContainer") -> None:
        return None


class FakeNetworkManager:
    def __init__(self) -> None:
        self._networks: Dict[str, FakeNetwork] = {}

    def get(self, name: str) -> FakeNetwork:
        if name not in self._networks:
            raise NotFound("network not found")
        return self._networks[name]

    def create(self, name: str, **_: Any) -> FakeNetwork:
        network = FakeNetwork(name)
        self._networks[name] = network
        return network


class FakeContainer:
    def __init__(self) -> None:
        self.status = "running"
        self.attrs = {"NetworkSettings": {"Networks": {"sandbox-net": {}}}}
        self.commands: List[list[str] | str] = []

    def exec_run(self, command: list[str] | str, stdout: bool = True, stderr: bool = True):
        self.commands.append(command)
        cmd_list = command if isinstance(command, list) else [command]
        if "has-session" in cmd_list:
            return FakeExecResult(exit_code=1, output="")
        if "capture-pane" in cmd_list:
            return FakeExecResult(output="pane output\n[sandbox]$ ")
        return FakeExecResult(output="ok\n[sandbox]$ ")

    def start(self) -> None:
        self.status = "running"

    def reload(self) -> None:
        return None


class FakeContainerManager:
    def __init__(self, container: FakeContainer) -> None:
        self._container = container
        self._exists = False
        self.last_run_args: List[Any] = []
        self.last_run_kwargs: Dict[str, Any] = {}

    def get(self, name: str) -> FakeContainer:
        if not self._exists:
            raise NotFound("container not found")
        return self._container

    def run(self, *args: Any, **kwargs: Any) -> FakeContainer:
        self._exists = True
        self.last_run_args = list(args)
        self.last_run_kwargs = dict(kwargs)
        return self._container


class FakeDockerClient:
    def __init__(self) -> None:
        self._container = FakeContainer()
        self.networks = FakeNetworkManager()
        self.containers = FakeContainerManager(self._container)


def _patched_sandbox(
    monkeypatch: pytest.MonkeyPatch,
    scope: Optional[List[str]] = None,
) -> tools.DockerSandbox:
    client = FakeDockerClient()
    monkeypatch.setattr(tools.docker, "from_env", lambda: client)
    return tools.DockerSandbox(scope=scope or ["10.0.0.1"])


def _mock_dns(ip: str, family: int = socket.AF_INET):
    def _resolver(*_args: Any, **_kwargs: Any):
        if family == socket.AF_INET6:
            return [(family, socket.SOCK_STREAM, 6, "", (ip, 0, 0, 0))]
        return [(family, socket.SOCK_STREAM, 6, "", (ip, 0))]

    return _resolver


def test_tmux_prompt_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox = _patched_sandbox(monkeypatch)
    output = sandbox.run_scoped_command("echo test", targets=["10.0.0.1"])
    assert "[sandbox]$" in output
    assert any(
        isinstance(cmd, list) and "export PS1" in " ".join(cmd)
        for cmd in sandbox.container.commands
    )


def test_scope_violation_blocks_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox = _patched_sandbox(monkeypatch)
    with pytest.raises(tools.ScopeViolation):
        sandbox.run_scoped_command("echo test", targets=["192.168.1.10"])


def test_scope_violation_blocks_out_of_scope_hostname(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = _patched_sandbox(monkeypatch, scope=["10.0.0.0/24"])
    monkeypatch.setattr(tools.socket, "getaddrinfo", _mock_dns("203.0.113.10"))
    with pytest.raises(tools.ScopeViolation):
        sandbox.run_scoped_command("echo test", targets=["evil.example"])


def test_scope_violation_blocks_invalid_cidr_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = _patched_sandbox(monkeypatch, scope=["10.0.0.0/24"])
    with pytest.raises(tools.ScopeViolation):
        sandbox.run_scoped_command("echo test", targets=["10.0.0.0/33"])


def test_scope_violation_blocks_embedded_out_of_scope_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = _patched_sandbox(monkeypatch, scope=["192.168.1.0/24"])
    with pytest.raises(tools.ScopeViolation):
        sandbox.run_scoped_command(
            "curl http://192.168.2.10", targets=["192.168.1.10"]
        )


def test_scope_violation_blocks_out_of_scope_ipv6(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = _patched_sandbox(monkeypatch, scope=["2001:db8::/64"])
    with pytest.raises(tools.ScopeViolation):
        sandbox.run_scoped_command("echo test", targets=["2001:db8:1::1"])


def test_container_hardening_flags_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox = _patched_sandbox(monkeypatch)
    run_kwargs = sandbox.client.containers.last_run_kwargs

    assert run_kwargs["read_only"] is True
    assert "/workspace" in run_kwargs["tmpfs"]
    assert "noexec" in run_kwargs["tmpfs"]["/workspace"]
    assert run_kwargs["mem_limit"] == sandbox.config.memory_limit
    assert run_kwargs["nano_cpus"] == int(sandbox.config.cpu_limit * 1_000_000_000)
    assert run_kwargs["pids_limit"] == sandbox.config.pids_limit
    assert run_kwargs["user"] == sandbox.config.run_as_user
    security_opts = run_kwargs["security_opt"]
    assert "no-new-privileges:true" in security_opts
    assert any(opt.startswith("seccomp=") for opt in security_opts)
