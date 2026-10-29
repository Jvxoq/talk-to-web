"""Value objects for the chat context: the rules for reading what a user said."""

import ipaddress
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from app.domain.chat.errors import EmptyUserMessage, UnsafeUrl

_URL_PATTERN = re.compile(r"https?://[^\s]+")

# Sentence punctuation stuck to the end of a URL.
_URL_TRAILING_NOISE = ".,;:!?\"'"

# A closing bracket is noise only when unbalanced. Wikipedia URLs like
# "/wiki/Hexagonal_architecture_(software)" need theirs.
_URL_BRACKET_PAIRS = {")": "(", "]": "[", "}": "{", ">": "<"}


def _trim_trailing_noise(url: str) -> str:
    """Drop the sentence punctuation a URL collected, keeping balanced brackets."""
    while url:
        last = url[-1]
        if last in _URL_TRAILING_NOISE:
            url = url[:-1]
            continue

        opener = _URL_BRACKET_PAIRS.get(last)
        if opener is not None and url.count(last) > url.count(opener):
            url = url[:-1]
            continue

        break
    return url


@dataclass(frozen=True, slots=True)
class UserMessage:
    """One thing the user typed, with the rules for reading it."""

    text: str

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise EmptyUserMessage()

    def urls(self) -> tuple[str, ...]:
        """Every http(s) address mentioned, in the order they were written."""
        found = (_trim_trailing_noise(url) for url in _URL_PATTERN.findall(self.text))
        return tuple(url for url in found if url)


_FETCHABLE_SCHEMES = frozenset({"http", "https"})
_DEFAULT_PORTS = {"http": 80, "https": 443}


def is_blocked_address(ip: str) -> bool:
    """Whether an IP is off-limits. Only public internet addresses are allowed.

    Private ranges reach internal services. 169.254.169.254 hands out cloud credentials.
    """
    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        # Not an address. Refuse rather than assume safe.
        return True

    # Unwrap IPv4-mapped IPv6, which the v6 checks would miss.
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped

    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


@dataclass(frozen=True, slots=True)
class FetchableUrl:
    """A URL the server may open, checked without network access.

    The adapter still resolves the host. Ports are not restricted.
    """

    value: str
    host: str
    port: int

    @classmethod
    def parse(cls, raw: str) -> "FetchableUrl":
        parts = urlsplit(raw)

        scheme = parts.scheme.lower()
        if scheme not in _FETCHABLE_SCHEMES:
            raise UnsafeUrl(raw, f"scheme {scheme or 'missing'!r} is not http or https")

        try:
            hostname = parts.hostname
            port = parts.port
        except ValueError as error:
            # urlsplit raises on a bad port only when accessed.
            raise UnsafeUrl(raw, "malformed host or port") from error

        if not hostname:
            raise UnsafeUrl(raw, "no host")

        # A literal IP needs no DNS, so it is checked here.

        if _looks_like_ip(hostname) and is_blocked_address(hostname):
            raise UnsafeUrl(raw, f"address {hostname} is not on the public internet")

        return cls(value=raw, host=hostname, port=port or _DEFAULT_PORTS[scheme])


def _looks_like_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True
