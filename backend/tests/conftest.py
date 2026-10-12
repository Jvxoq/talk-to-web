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

    Each attempt raises, and the test also fails at teardown, because code
    under test with a broad `except` can swallow the raise.

    Loopback stays open on purpose: the scraper suite serves real pages from
    a local `aiohttp.web` server, and those behaviours only show over a real
    socket. Nothing past loopback is reachable, so no test can spend money or
    leak a key off the box.

    The integration suite stands down from this gate: it exists to speak to
    real Postgres and Qdrant, and it only runs when asked with `-m integration`.
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
