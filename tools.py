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

import functools
import ipaddress
import re
import shlex
import socket
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set
from urllib.parse import urlparse

import docker
from docker.errors import NotFound
try:
    from langsmith import traceable
except ImportError:  # pragma: no cover - optional dependency
    def traceable(func=None, **_kwargs):
        def decorator(wrapped):
            return functools.wraps(wrapped)(wrapped)

        if func is None:
            return decorator
        return decorator(func)


class ScopeViolation(RuntimeError):
    """Raised when a command attempts to access a target outside authorized scope."""


_IPV4_CIDR_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b")
_IPV6_CIDR_PATTERN = re.compile(r"\b(?:[A-Fa-f0-9]{0,4}:){2,7}[A-Fa-f0-9]{0,4}(?:/\d{1,3})?\b")
_HOSTNAME_PATTERN = re.compile(
    r"^(?=.{1,253}$)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z0-9-]{2,63}$"
)


def _strip_wrapping(value: str) -> str:
    """
    Remove wrapping punctuation around tokens (URLs, brackets, etc.).
    """

    return value.strip("[](){}<>,;\"'")


def _looks_like_hostname(value: str) -> bool:
    """
    Heuristic for hostname detection without over-matching flags/verbs.
    """

    if value == "localhost":
        return True
    if "." not in value:
        return False
    return bool(_HOSTNAME_PATTERN.match(value))


def _resolve_hostname(
    hostname: str,
) -> Set[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """
    Resolve hostnames to IP addresses for scope enforcement.
    """

    try:
        info = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ScopeViolation(
            f"Hostname {hostname} could not be resolved for scope validation."
        ) from exc

    addresses: Set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
    for _, _, _, _, sockaddr in info:
        ip_text = sockaddr[0]
        if "%" in ip_text:
            ip_text = ip_text.split("%", 1)[0]
        try:
            addresses.add(ipaddress.ip_address(ip_text))
        except ValueError:
            continue
    if not addresses:
        raise ScopeViolation(
            f"Hostname {hostname} resolved to no valid IP addresses."
        )
    return addresses


def _normalize_scope(
    scope: Sequence[str],
) -> List[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    """
    Normalize a list of IPs/subnets/hostnames into ipaddress network objects.
    """

    networks: List[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for entry in scope:
        entry = entry.strip()
        if not entry:
            continue
        # Convert plain IPs to /32 or /128 networks for consistent checks.
        if "/" not in entry:
            try:
                ip = ipaddress.ip_address(entry)
            except ValueError:
                for resolved in _resolve_hostname(entry):
                    networks.append(
                        ipaddress.ip_network(
                            f"{resolved}/{resolved.max_prefixlen}", strict=False
                        )
                    )
            else:
                networks.append(
                    ipaddress.ip_network(f"{ip}/{ip.max_prefixlen}", strict=False)
                )
        else:
            try:
                networks.append(ipaddress.ip_network(entry, strict=False))
            except ValueError as exc:
                raise ScopeViolation(f"Invalid scope entry {entry}.") from exc
    return networks


def _extract_command_targets(command: str) -> Set[str]:
    """
    Extract IPs, CIDRs, and hostnames from a command string.
    """

    targets: Set[str] = set()

    for match in _IPV4_CIDR_PATTERN.findall(command):
        targets.add(match)
    for match in _IPV6_CIDR_PATTERN.findall(command):
        targets.add(match)

    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()

    for token in tokens:
        raw_token = token
        token = _strip_wrapping(token)
        if not token or token.startswith("-"):
            continue

        if "://" in token:
            parsed = urlparse(token)
            if parsed.hostname:
                targets.add(parsed.hostname)
                continue

        if token.count(":") == 1 and not token.startswith("["):
            parsed = urlparse(f"tcp://{token}")
            if parsed.hostname:
                targets.add(parsed.hostname)
                continue

        if raw_token.startswith("[") and "]" in raw_token:
            bracketed = raw_token[1 : raw_token.index("]")]
            if bracketed:
                targets.add(bracketed)
                continue

        if _looks_like_hostname(token):
            targets.add(token)

    return targets


def _targets_in_scope(targets: Iterable[str], scope: Sequence[str]) -> None:
    """
    Validate that all targets reside inside the authorized scope.
    """

    networks = _normalize_scope(scope)
    if not networks:
        raise ScopeViolation("No authorized scope configured.")

    for target in targets:
        if not target:
            continue
        if "/" in target:
            try:
                target_network = ipaddress.ip_network(target, strict=False)
            except ValueError as exc:
                raise ScopeViolation(f"Invalid CIDR target {target}.") from exc

            same_family = [net for net in networks if net.version == target_network.version]
            if not same_family or not any(
                target_network.subnet_of(network) for network in same_family
            ):
                raise ScopeViolation(
                    f"Target {target} is outside the authorized engagement scope."
                )
            continue

        try:
            addresses = {ipaddress.ip_address(target)}
        except ValueError:
            addresses = _resolve_hostname(target)

        for ip in addresses:
            same_family = [net for net in networks if net.version == ip.version]
            if not same_family or not any(ip in network for network in same_family):
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
    error_output_limit: int = 500
    memory_limit: str = "1g"
    cpu_limit: float = 1.0
    pids_limit: int = 256
    read_only_rootfs: bool = True
    workspace_dir: str = "/workspace"
    workspace_tmpfs_size: str = "256m"
    run_as_user: str = "1000:1000"
    seccomp_profile: str = "default"


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
                user=self.config.run_as_user,
                working_dir=self.config.workspace_dir,
                read_only=self.config.read_only_rootfs,
                tmpfs={
                    self.config.workspace_dir: (
                        f"rw,noexec,nosuid,size={self.config.workspace_tmpfs_size}"
                    )
                },
                mem_limit=self.config.memory_limit,
                nano_cpus=self._nano_cpus(),
                pids_limit=self.config.pids_limit,
                security_opt=self._security_opts(),
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

    def _exec(self, command: list[str] | str) -> str:
        """
        Execute a command inside the container and return stdout.
        """

        result = self.container.exec_run(command, stdout=True, stderr=True)
        output = result.output.decode(errors="replace")
        if result.exit_code != 0:
            snippet = output[-self.config.error_output_limit:] if output else ""
            raise RuntimeError(
                f"Sandbox command failed. Output (truncated): {snippet}"
            )
        return output

    def _exec_shell(self, command: str) -> str:
        """
        Execute a shell command inside the container (for controlled internal use).
        """

        return self._exec(["/bin/sh", "-c", command])

    def _ensure_tmux_available(self) -> None:
        """
        Guarantee tmux is installed, since all persistent sessions rely on it.
        """

        try:
            self._exec(["tmux", "-V"])
        except RuntimeError as exc:
            raise RuntimeError(
                "tmux is not available in the sandbox image. "
                "Bake tmux into the Kali image before running the sandbox."
            ) from exc

    def _nano_cpus(self) -> int:
        """
        Convert CPU limit to Docker nano_cpus units.
        """

        nano_cpus = int(self.config.cpu_limit * 1_000_000_000)
        return nano_cpus if nano_cpus > 0 else 1_000_000_000

    def _security_opts(self) -> list[str]:
        """
        Build security options for container hardening.
        """

        seccomp_profile = self.config.seccomp_profile or "default"
        return ["no-new-privileges:true", f"seccomp={seccomp_profile}"]

    @staticmethod
    def _sanitize_session(session: str) -> str:
        """
        Constrain tmux session names to safe characters.
        """

        return re.sub(r"[^A-Za-z0-9_.-]", "_", session)

    def _ensure_tmux_session(self, session: str) -> None:
        """
        Create the tmux session if it does not exist and seed a known prompt.
        """

        session = self._sanitize_session(session)
        tmux = ["tmux", "-L", self.config.tmux_socket_name]
        has_session = self.container.exec_run(tmux + ["has-session", "-t", session])
        if has_session.exit_code != 0:
            self._exec(tmux + ["new-session", "-d", "-s", session])

        # Standardize the shell prompt for reliable prompt detection.
        prompt_command = "export PS1='[sandbox]$ '"
        self._exec(tmux + ["send-keys", "-t", session, prompt_command, "C-m"])

    def _wait_for_prompt(self, session: str, timeout: int) -> str:
        """
        Wait until the expected prompt appears in the tmux buffer.
        """

        tmux = ["tmux", "-L", self.config.tmux_socket_name]
        deadline = time.time() + timeout
        buffer_output = ""
        while time.time() < deadline:
            buffer_output = self._exec(
                tmux + ["capture-pane", "-pt", session, "-S", "-200"]
            )
            if re.search(self.config.prompt_regex, buffer_output):
                return buffer_output
            time.sleep(0.5)
        raise TimeoutError("Timed out waiting for sandbox prompt.")

    @traceable(run_type="tool", name="tool.run_scoped_command")
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

        # Double-check any embedded targets (IPs, CIDRs, hostnames) to prevent violations.
        for target in _extract_command_targets(command):
            _targets_in_scope([target], self.scope)

        self._ensure_tmux_session(session)
        session = self._sanitize_session(session)
        tmux = ["tmux", "-L", self.config.tmux_socket_name]
        self._exec(tmux + ["send-keys", "-t", session, command, "C-m"])
        return self._wait_for_prompt(
            session, timeout or self.config.command_timeout_seconds
        )

    @traceable(run_type="tool", name="tool.send_interactive")
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
        session = self._sanitize_session(session)
        tmux = ["tmux", "-L", self.config.tmux_socket_name]
        self._exec(tmux + ["send-keys", "-t", session, keystrokes])
        return self._wait_for_prompt(
            session, timeout or self.config.command_timeout_seconds
        )


class ScopedToolset:
    """
    High-level interface for agents. All commands pass through scope checks.
    """

    def __init__(self, sandbox: DockerSandbox):
        self.sandbox = sandbox

    @traceable(run_type="tool", name="tool.nmap_scan")
    def nmap_scan(self, targets: Sequence[str], options: str = "-sV -T3") -> str:
        """
        Run an Nmap scan in the sandbox with strict scope enforcement.
        """

        target_list = " ".join(targets)
        command = f"nmap {options} {target_list}"
        return self.sandbox.run_scoped_command(command, targets, session="recon")

    @traceable(run_type="tool", name="tool.execute_payload")
    def execute_payload(self, command: str, targets: Sequence[str]) -> str:
        """
        Execute an exploit payload in the sandbox.
        """

        return self.sandbox.run_scoped_command(command, targets, session="exploit")
