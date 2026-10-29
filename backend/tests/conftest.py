import socket
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NoReturn

import pytest

# Make `app` and `tests` importable when pytest is run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _is_loopback(host: object) -> bool:
    return isinstance(host, str) and host.lower() in _LOOPBACK_HOSTS


@pytest.fixture(autouse=True)
def no_real_network(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Iterator[None]:
    """Fail any test that reaches past loopback or resolves a real name.

    Also fails at teardown, in case a broad `except` swallowed the raise.
    Loopback stays open for the scraper's local server. Integration tests skip
    this gate.
    """

    if request.node.get_closest_marker("integration") is not None:
        yield
        return

    attempts: list[str] = []
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def refuse(host: object) -> NoReturn:
        attempts.append(repr(host))
        raise AssertionError(f"a test reached the real network: {host!r}")

    def connect(self: socket.socket, address: Any) -> None:
        host = address[0] if isinstance(address, tuple) else address
        if _is_loopback(host):
            real_connect(self, address)
            return
        refuse(host)

    def connect_ex(self: socket.socket, address: Any) -> int:
        host = address[0] if isinstance(address, tuple) else address
        if _is_loopback(host):
            return real_connect_ex(self, address)
        refuse(host)

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if _is_loopback(host):
            return real_getaddrinfo(host, *args, **kwargs)
        refuse(host)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    yield
    assert attempts == [], f"a test reached the real network: {attempts}"
