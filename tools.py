"""
tools.py
--------
Execution environment and security controls for the red teaming system.

Key requirements implemented here:
1) Strict two-network Docker layout (management plane vs. sandbox).
2) Scoped, declarative permissions that mathematically block out-of-scope targets.
3) Persistent tmux sessions inside the sandbox with prompt detection.
4) No direct host execution — all offensive tooling runs in the Kali container.
"""

from __future__ import annotations

import ipaddress
import re
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set

import docker
from docker.errors import NotFound


class ScopeViolation(RuntimeError):
    """Raised when a command attempts to access a target outside authorized scope."""


def _normalize_scope(
    scope: Sequence[str],
) -> List[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    """
    Normalize a list of IPs/subnets into ipaddress network objects.
    """

    networks: List[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for entry in scope:
        # Convert plain IPs to /32 or /128 networks for consistent checks.
        if "/" not in entry:
            ip = ipaddress.ip_address(entry)
            networks.append(
                ipaddress.ip_network(f"{ip}/{ip.max_prefixlen}", strict=False)
            )
        else:
            networks.append(ipaddress.ip_network(entry, strict=False))
    return networks


def _extract_ipv4_targets(command: str) -> Set[str]:
    """
    Extract IPv4 addresses from a command string to guard against hidden targets.
    """

    return set(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", command))


def _targets_in_scope(targets: Iterable[str], scope: Sequence[str]) -> None:
    """
    Validate that all targets reside inside the authorized scope.
    """

    networks = _normalize_scope(scope)
    for target in targets:
        if "/" in target:
            target_network = ipaddress.ip_network(target, strict=False)
            if not any(target_network.subnet_of(network) for network in networks):
                raise ScopeViolation(
                    f"Target {target} is outside the authorized engagement scope."
                )
        else:
            ip = ipaddress.ip_address(target)
            if not any(ip in network for network in networks):
                raise ScopeViolation(
                    f"Target {target} is outside the authorized engagement scope."
                )


@dataclass(frozen=True)
class SandboxConfig:
    """
    Immutable configuration for the Docker-based sandbox.
    """

    kali_image: str = "kalilinux/kali-rolling"
    container_name: str = "kali-sandbox"
    management_network: str = "decepticon-net"
    sandbox_network: str = "sandbox-net"
    tmux_socket_name: str = "redteam-sandbox"
    prompt_regex: str = r"\[sandbox\]\$ "
    command_timeout_seconds: int = 300


class DockerSandbox:
    """
    Secure execution wrapper that runs offensive tooling in an isolated Kali container.

    The orchestration logic runs in the management plane; the Kali container is
    attached only to the sandbox network to enforce isolation.
    """

    def __init__(self, scope: Sequence[str], config: Optional[SandboxConfig] = None):
        self.scope = list(scope)
        self.config = config or SandboxConfig()
        self.client = docker.from_env()
        self._ensure_networks()
        self.container = self._ensure_container()
        self._ensure_tmux_available()

    def _ensure_networks(self) -> None:
        """
        Create the management and sandbox networks if they do not already exist.
        """

        for name, internal in [
            (self.config.management_network, False),
            (self.config.sandbox_network, True),
        ]:
            try:
                self.client.networks.get(name)
            except NotFound:
                self.client.networks.create(
                    name,
                    driver="bridge",
                    internal=internal,
                    attachable=False,
                    check_duplicate=True,
                )

    def _ensure_container(self):
        """
        Ensure the Kali container exists and is attached solely to the sandbox network.
        """

        try:
            container = self.client.containers.get(self.config.container_name)
        except NotFound:
            container = self.client.containers.run(
                self.config.kali_image,
                name=self.config.container_name,
                command="sleep infinity",
                detach=True,
                tty=True,
                stdin_open=True,
                network=self.config.sandbox_network,
                security_opt=["no-new-privileges:true"],
            )
        else:
            if container.status != "running":
                container.start()

        # Ensure the container is not attached to any other networks.
        container.reload()
        for network_name in list(container.attrs["NetworkSettings"]["Networks"].keys()):
            if network_name != self.config.sandbox_network:
                self.client.networks.get(network_name).disconnect(container)
        return container

    def _exec(self, command: str) -> str:
        """
        Execute a command inside the container and return stdout.
        """

        result = self.container.exec_run(command, stdout=True, stderr=True)
        output = result.output.decode(errors="replace")
        if result.exit_code != 0:
            raise RuntimeError(f"Sandbox command failed: {command}\n{output}")
        return output

    def _ensure_tmux_available(self) -> None:
        """
        Guarantee tmux is installed, since all persistent sessions rely on it.
        """

        try:
            self._exec("tmux -V")
        except RuntimeError:
            # Kali images typically include tmux; if not, install it.
            self._exec("apt-get update && apt-get install -y tmux")

    def _ensure_tmux_session(self, session: str) -> None:
        """
        Create the tmux session if it does not exist and seed a known prompt.
        """

        tmux = f"tmux -L {self.config.tmux_socket_name}"
        has_session = self.container.exec_run(
            f"{tmux} has-session -t {session}"
        )
        if has_session.exit_code != 0:
            self._exec(f"{tmux} new-session -d -s {session}")

        # Standardize the shell prompt for reliable prompt detection.
        prompt_command = f"export PS1='[sandbox]$ '"
        self._exec(f"{tmux} send-keys -t {session} \"{prompt_command}\" C-m")

    def _wait_for_prompt(self, session: str, timeout: int) -> str:
        """
        Wait until the expected prompt appears in the tmux buffer.
        """

        tmux = f"tmux -L {self.config.tmux_socket_name}"
        deadline = time.time() + timeout
        buffer_output = ""
        while time.time() < deadline:
            buffer_output = self._exec(
                f"{tmux} capture-pane -pt {session} -S -200"
            )
            if re.search(self.config.prompt_regex, buffer_output):
                return buffer_output
            time.sleep(0.5)
        raise TimeoutError("Timed out waiting for sandbox prompt.")

    def run_scoped_command(
        self,
        command: str,
        targets: Sequence[str],
        session: str = "default",
        timeout: Optional[int] = None,
    ) -> str:
        """
        Run a command inside the sandbox after enforcing target scope validation.
        """

        _targets_in_scope(targets, self.scope)

        # Double-check any IPs embedded in the command to prevent stealth violations.
        for ip in _extract_ipv4_targets(command):
            _targets_in_scope([ip], self.scope)

        self._ensure_tmux_session(session)
        tmux = f"tmux -L {self.config.tmux_socket_name}"
        self._exec(f"{tmux} send-keys -t {session} \"{command}\" C-m")
        return self._wait_for_prompt(
            session, timeout or self.config.command_timeout_seconds
        )

    def send_interactive(
        self,
        session: str,
        keystrokes: str,
        timeout: Optional[int] = None,
    ) -> str:
        """
        Send follow-up keystrokes to an existing tmux session to handle prompts.
        """

        self._ensure_tmux_session(session)
        tmux = f"tmux -L {self.config.tmux_socket_name}"
        self._exec(f"{tmux} send-keys -t {session} \"{keystrokes}\"")
        return self._wait_for_prompt(
            session, timeout or self.config.command_timeout_seconds
        )


class ScopedToolset:
    """
    High-level interface for agents. All commands pass through scope checks.
    """

    def __init__(self, sandbox: DockerSandbox):
        self.sandbox = sandbox

    def nmap_scan(self, targets: Sequence[str], options: str = "-sV -T3") -> str:
        """
        Run an Nmap scan in the sandbox with strict scope enforcement.
        """

        target_list = " ".join(targets)
        command = f"nmap {options} {target_list}"
        return self.sandbox.run_scoped_command(command, targets, session="recon")

    def execute_payload(self, command: str, targets: Sequence[str]) -> str:
        """
        Execute an exploit payload in the sandbox.
        """

        return self.sandbox.run_scoped_command(command, targets, session="exploit")
