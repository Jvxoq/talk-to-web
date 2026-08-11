"""Value objects for the chat context: the rules for reading what a user said.

Composing the model's prompt used to live here too. It does not any more: the
agent builds a message list and lets the model decide what context it needs,
so there is no single "final text" left to assemble.
"""

import ipaddress
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from app.domain.chat.errors import EmptyUserMessage, UnsafeUrl

_URL_PATTERN = re.compile(r"https?://[^\s]+")

# Trailing punctuation is almost always sentence punctuation rather than part of
# the address, and a stray "." or "," turns a good fetch into a 404.
_URL_TRAILING_NOISE = ".,;:!?\"'"

# A closing bracket is the exception: it may genuinely belong to the address.
# Wikipedia disambiguates titles with them - "/wiki/Hexagonal_architecture_(software)"
# - and stripping that ")" unconditionally requested a page that does not exist,
# so every such link silently contributed nothing. A closer is only noise when
# nothing earlier in the URL opened it.
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
    """
    Whether an IP is somewhere the server must never be talked into reaching.

    "Public internet only" is the whole rule. Everything else an attacker gets
    by naming an address is infrastructure that trusts anything already inside
    the network: 169.254.169.254 hands out cloud credentials, 127.0.0.1 and the
    private ranges reach Qdrant, Postgres and every other unauthenticated
    service on this host or in this VPC.
    """
    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        # Not an address at all. The caller resolves hostnames before asking,
        # so anything unparseable here is refused rather than assumed safe.
        return True

    # "::ffff:10.0.0.1" is 10.0.0.1 wearing an IPv6 costume, and none of the
    # v6 predicates below would call it private.
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
    """
    A URL the server is allowed to open, once it has been read carefully.

    Everything checkable without touching the network is checked here; the
    hostname still has to be resolved, which is I/O and therefore an adapter's
    job. Ports are deliberately not restricted - a legitimate article can live
    on :8080, and the address is what decides safety, not the port.
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
            # urlsplit defers parsing the authority, so a malformed port only
            # blows up on access.
            raise UnsafeUrl(raw, "malformed host or port") from error

        if not hostname:
            raise UnsafeUrl(raw, "no host")

        # A literal address skips name resolution entirely, so it is the one
        # case the domain can settle on its own.
        if _looks_like_ip(hostname) and is_blocked_address(hostname):
            raise UnsafeUrl(raw, f"address {hostname} is not on the public internet")

        return cls(value=raw, host=hostname, port=port or _DEFAULT_PORTS[scheme])


def _looks_like_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True
