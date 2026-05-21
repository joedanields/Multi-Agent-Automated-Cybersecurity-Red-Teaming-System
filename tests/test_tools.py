from __future__ import annotations

from typing import Any, Dict, List

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

    def get(self, name: str) -> FakeContainer:
        if not self._exists:
            raise NotFound("container not found")
        return self._container

    def run(self, *args: Any, **kwargs: Any) -> FakeContainer:
        self._exists = True
        return self._container


class FakeDockerClient:
    def __init__(self) -> None:
        self._container = FakeContainer()
        self.networks = FakeNetworkManager()
        self.containers = FakeContainerManager(self._container)


def _patched_sandbox(monkeypatch: pytest.MonkeyPatch) -> tools.DockerSandbox:
    client = FakeDockerClient()
    monkeypatch.setattr(tools.docker, "from_env", lambda: client)
    return tools.DockerSandbox(scope=["10.0.0.1"])


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
