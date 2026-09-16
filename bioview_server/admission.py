"""Who is allowed to connect, and how the answer is obtained."""

import contextlib
import ipaddress
import os
import socket
import sys
import threading

from bioview_common import get_local_addresses, is_local_request, log_print


POLICIES = ("loopback", "local", "ask", "any")

DEFAULT_POLICY = "ask"

PROMPT_TIMEOUT_S = 120


class AdmissionError(Exception):
    """The policy name given is not one this server knows."""


def normalize_policy(policy: str) -> str:
    name = str(policy or "").strip().lower()
    if name not in POLICIES:
        raise AdmissionError(
            f"Unknown connection policy {policy!r}; expected one of "
            f"{', '.join(POLICIES)}"
        )
    return name


def _describe(peer_ip: str, client_info: dict) -> str:
    """A one-line identification of whoever is knocking."""
    info = client_info or {}
    hostname = (info.get("hostname") or "").strip()
    name = (info.get("name") or "").strip()
    version = (info.get("version") or "").strip()

    who = hostname or name or "an unidentified BioView client"
    parts = [f"{who} ({peer_ip})"]
    if version:
        parts.append(f"BioView {version}")
    return ", ".join(parts)


class AdmissionController:
    """Decides whether a peer may open a session, asking a human if need be."""

    def __init__(self, policy: str = DEFAULT_POLICY, trusted=None, logger=None):
        self.policy = normalize_policy(policy)
        self.logger = logger

        self._decisions: dict[str, bool] = {}
        for address in trusted or ():
            address = str(address).strip()
            if address:
                self._decisions[address] = True

        self._prompt_lock = threading.Lock()

    def is_own_machine(self, peer_ip: str) -> bool:
        """True when the peer is this host: loopback, or one of our own NICs."""
        if not peer_ip:
            return False
        if peer_ip in get_local_addresses():
            return True
        with contextlib.suppress(ValueError):
            return ipaddress.ip_address(peer_ip).is_loopback
        return False

    def decide(self, peer_ip: str, client_info: dict = None) -> tuple[bool, str]:
        """Return ``(allowed, reason)`` for one incoming connection."""
        if not peer_ip:
            return False, "the connection had no usable peer address"

        if self.is_own_machine(peer_ip):
            return True, "same machine"

        if self.policy == "any":
            return True, "policy allows any address"

        if self.policy == "loopback":
            return False, (
                "this server only accepts connections from its own machine "
                "(--allow loopback)"
            )

        if self.policy == "local":
            if is_local_request(peer_ip):
                return True, "private address, policy allows the local network"
            return False, (
                f"{peer_ip} is not on this machine or a private network, and "
                "this server is running with --allow local"
            )

        remembered = self._decisions.get(peer_ip)
        if remembered is not None:
            return remembered, (
                "allowed earlier in this session"
                if remembered
                else "refused earlier in this session"
            )

        with self._prompt_lock:
            remembered = self._decisions.get(peer_ip)
            if remembered is not None:
                return remembered, (
                    "allowed earlier in this session"
                    if remembered
                    else "refused earlier in this session"
                )

            allowed = self._ask(peer_ip, client_info)
            self._decisions[peer_ip] = allowed

        return allowed, (
            "allowed by the operator at the server"
            if allowed
            else "the operator at the server refused this connection"
        )

    def may_probe(self, peer_ip: str) -> bool:
        """Whether a peer may be answered a discovery probe."""
        if self.is_own_machine(peer_ip):
            return True
        if self.policy in ("any", "ask"):
            return True
        if self.policy == "local":
            return is_local_request(peer_ip)
        return False

    def forget(self, peer_ip: str = None):
        """Drop remembered answers, so the next connection asks again."""
        if peer_ip is None:
            self._decisions.clear()
        else:
            self._decisions.pop(peer_ip, None)

    def _ask(self, peer_ip: str, client_info: dict) -> bool:
        """Put the question to whoever is at the server, however we can."""
        who = _describe(peer_ip, client_info)
        log_print(self.logger, "info", f"Connection request from {who}; asking...")

        for prompt in (self._ask_desktop, self._ask_console):
            answer = prompt(who, peer_ip)
            if answer is not None:
                log_print(
                    self.logger,
                    "info",
                    f"{who} was {'allowed' if answer else 'refused'}",
                )
                return answer

        log_print(
            self.logger,
            "warning",
            f"Refusing {who}: there is no way to ask at this server (no desktop "
            "and no console). Start it with --allow any or --trust "
            f"{peer_ip} to admit it without asking.",
        )
        return False

    def _question(self, who: str) -> tuple[str, str]:
        return (
            "Allow this computer to connect to BioView?",
            f"{who}\n\nwants to connect to the BioView server on this machine "
            "and drive its hardware.\n\nAllow the connection?",
        )

    def _ask_desktop(self, who: str, peer_ip: str):
        """A native dialog, or None when this machine has no desktop to show one on."""
        title, body = self._question(who)

        if os.name == "nt":
            return _ask_windows_messagebox(title, body, self.logger)
        return _ask_tk(title, body, self.logger)

    def _ask_console(self, who: str, peer_ip: str):
        """A console question, or None when nobody is at a terminal."""
        try:
            if not (sys.stdin and sys.stdin.isatty()):
                return None
        except (AttributeError, ValueError, OSError):
            return None

        stream = sys.stderr if sys.stderr else sys.stdout
        try:
            print(
                f"\n{who}\nwants to connect to this BioView server.",
                file=stream,
                flush=True,
            )
            answer = input(f"Allow {peer_ip}? [y/N] ")
        except (EOFError, KeyboardInterrupt, OSError):
            return None
        return str(answer).strip().lower() in ("y", "yes")


_MB_YESNO = 0x00000004
_MB_ICONQUESTION = 0x00000020
_MB_DEFBUTTON2 = 0x00000100
_MB_SETFOREGROUND = 0x00010000
_MB_TOPMOST = 0x00040000
_IDYES = 6
_IDTIMEOUT = 32000


def _ask_windows_messagebox(title: str, body: str, logger=None):
    """Windows: a plain user32 message box. None if it could not be shown."""
    flags = (
        _MB_YESNO | _MB_ICONQUESTION | _MB_DEFBUTTON2 | _MB_SETFOREGROUND | _MB_TOPMOST
    )
    try:
        import ctypes

        user32 = ctypes.windll.user32
        try:
            result = user32.MessageBoxTimeoutW(
                None, body, title, flags, 0, int(PROMPT_TIMEOUT_S * 1000)
            )
        except AttributeError:
            result = user32.MessageBoxW(None, body, title, flags)
    except Exception as e:
        log_print(logger, "debug", f"Could not show a connection prompt: {e}")
        return None
    if not result:
        return None
    if result == _IDTIMEOUT:
        log_print(
            logger,
            "warning",
            f"Nobody answered the connection prompt within {PROMPT_TIMEOUT_S:g}s",
        )
        return False
    return result == _IDYES


def _ask_tk(title: str, body: str, logger=None):
    """Linux/macOS: a Tk dialog. None when there is no display or no tkinter."""
    if os.name != "nt" and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        return None
    try:
        import tkinter
        from tkinter import messagebox

        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.after(int(PROMPT_TIMEOUT_S * 1000), root.destroy)
        try:
            return bool(messagebox.askyesno(title, body, parent=root))
        except Exception:
            return False
        finally:
            with contextlib.suppress(Exception):
                root.destroy()
    except Exception as e:
        log_print(logger, "debug", f"Could not show a connection prompt: {e}")
        return None


def resolve_hostnames(entries) -> set:
    """Turn ``--trust`` entries into addresses, accepting hostnames too."""
    addresses = set()
    for entry in entries or ():
        entry = str(entry).strip()
        if not entry:
            continue
        addresses.add(entry)
        with contextlib.suppress(ValueError):
            ipaddress.ip_address(entry)
            continue
        with contextlib.suppress(OSError):
            addresses.update(socket.gethostbyname_ex(entry)[2])
    return addresses
