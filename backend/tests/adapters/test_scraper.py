"""The aiohttp scraper against a real HTTP server.

Every test below makes an actual request over an actual socket to an actual
`aiohttp.web` application, and parses the result with actual BeautifulSoup and
lxml. Redirect following, the byte ceiling, charset handling and content-type
refusal are all things that only behave the way the code assumes when a real
server is on the other end, so none of them are stubbed.

The one thing that is: `is_blocked_address`, as the adapter sees it. The test
server necessarily lives on loopback, which the SSRF guard exists to refuse.
`treat_local_as_public` relaxes that one predicate on the adapter's module, and
only for the tests that need to reach the server. `TestRefusesPrivateTargets`
runs with the real predicate and is what proves the guard refuses; the domain's
own copy is never patched by anything here.
"""

import re
from collections.abc import AsyncIterator, Awaitable, Callable

import aiohttp
import pytest
from aiohttp import web
from aiohttp.typedefs import Handler

from app.adapters.web.aiohttp_scraper import (
    MAX_PAGE_BYTES,
    MAX_PAGE_CHARS,
    MAX_REDIRECTS,
    AiohttpWebContentFetcher,
)
from app.domain.chat.errors import UnsafeUrl

ARTICLE = "<html><body><article>The readable part.</article><nav>Skip me</nav></body></html>"

Serve = Callable[[web.Application], Awaitable[str]]
"""Start a test server for one application and hand back its base URL."""


@pytest.fixture
def treat_local_as_public(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the guard accept 127.0.0.1, so the test server is reachable.

    Patched on the scraper's module rather than at the domain, so the domain
    rule itself is never redefined - only this adapter's view of it, for the
    duration of one test.
    """
    monkeypatch.setattr(
        "app.adapters.web.aiohttp_scraper.is_blocked_address",
        lambda address: False,
    )


@pytest.fixture
async def serve() -> AsyncIterator[Serve]:
    """Start an `aiohttp.web` app on a free port and return its base URL."""
    sites: list[web.TCPSite] = []
    runners: list[web.AppRunner] = []

    async def start(application: web.Application) -> str:
        runner = web.AppRunner(application)
        await runner.setup()
        runners.append(runner)
        site = web.TCPSite(runner, "localhost", 0)
        await site.start()
        sites.append(site)
        port = next(iter(runner.addresses))[1]
        # A hostname, not `127.0.0.1`. A literal address is judged by the domain
        # in `FetchableUrl.parse`, before the adapter is involved at all, and
        # relaxing a domain rule from a test would be relaxing the wrong thing.
        # A name reaches the adapter's DNS check, which is what
        # `treat_local_as_public` speaks to.
        return f"http://localhost:{port}"

    try:
        yield start
    finally:
        for runner in runners:
            await runner.cleanup()


@pytest.fixture
async def fetcher() -> AsyncIterator[AiohttpWebContentFetcher]:
    # A short timeout so a hung server fails the test rather than the suite.
    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
    fetcher = AiohttpWebContentFetcher(session)
    try:
        yield fetcher
    finally:
        await fetcher.aclose()


def page(body: str, content_type: str = "text/html") -> Handler:
    async def handler(_request: web.Request) -> web.Response:
        return web.Response(body=body.encode(), content_type=content_type)

    return handler


class TestParsesRealPages:
    async def test_it_returns_the_readable_body_text(
        self,
        fetcher: AiohttpWebContentFetcher,
        serve: Serve,
        treat_local_as_public: None,
    ) -> None:
        application = web.Application()
        application.router.add_get("/", page(ARTICLE))
        base = await serve(application)

        assert await fetcher.fetch_all([f"{base}/"]) == "The readable part."

    @pytest.mark.parametrize(
        ("html", "expected"),
        [
            ("<div id='bodyContent'>Wikipedia body</div>", "Wikipedia body"),
            ("<main>Main element</main>", "Main element"),
            ("<article>Article element</article>", "Article element"),
            # No container at all: the fallback is the whole document, which is
            # what stops a plain page silently producing an empty string. The
            # legacy parser only looked for `div#bodyContent`, so every site but
            # Wikipedia returned nothing.
            ("<p>Just a paragraph</p>", "Just a paragraph"),
        ],
    )
    async def test_it_finds_the_content_element_on_pages_of_every_shape(
        self,
        html: str,
        expected: str,
        fetcher: AiohttpWebContentFetcher,
        serve: Serve,
        treat_local_as_public: None,
    ) -> None:
        application = web.Application()
        application.router.add_get("/", page(f"<html><body>{html}</body></html>"))
        base = await serve(application)

        assert await fetcher.fetch_all([f"{base}/"]) == expected

    async def test_a_page_with_no_text_yields_an_empty_string(
        self,
        fetcher: AiohttpWebContentFetcher,
        serve: Serve,
        treat_local_as_public: None,
    ) -> None:
        application = web.Application()
        application.router.add_get("/", page("<html><body><main></main></body></html>"))
        base = await serve(application)

        assert await fetcher.fetch_all([f"{base}/"]) == ""

    async def test_a_page_is_truncated_to_the_character_budget(
        self,
        fetcher: AiohttpWebContentFetcher,
        serve: Serve,
        treat_local_as_public: None,
    ) -> None:
        # One enormous document must not eat the model's whole context window,
        # nor the prompt budget belonging to the other URLs in the message.
        body = "x" * (MAX_PAGE_CHARS + 5_000)
        application = web.Application()
        application.router.add_get("/", page(f"<html><body><main>{body}</main></body></html>"))
        base = await serve(application)

        assert len(await fetcher.fetch_all([f"{base}/"])) == MAX_PAGE_CHARS

    async def test_a_declared_charset_is_honoured(
        self,
        fetcher: AiohttpWebContentFetcher,
        serve: Serve,
        treat_local_as_public: None,
    ) -> None:
        async def handler(_request: web.Request) -> web.Response:
            return web.Response(
                body="<html><body><main>café</main></body></html>".encode("latin-1"),
                content_type="text/html",
                charset="latin-1",
            )

        application = web.Application()
        application.router.add_get("/", handler)
        base = await serve(application)

        assert await fetcher.fetch_all([f"{base}/"]) == "café"

    async def test_an_encoding_python_has_never_heard_of_falls_back_to_utf8(
        self,
        fetcher: AiohttpWebContentFetcher,
        serve: Serve,
        treat_local_as_public: None,
    ) -> None:
        # A server is free to name any charset it likes. Raising `LookupError`
        # out of a fetch would lose the page over a header nobody controls.
        async def handler(_request: web.Request) -> web.Response:
            response = web.Response(body=b"<html><body><main>ok</main></body></html>")
            response.headers["Content-Type"] = "text/html; charset=invented-9"
            return response

        application = web.Application()
        application.router.add_get("/", handler)
        base = await serve(application)

        assert await fetcher.fetch_all([f"{base}/"]) == "ok"


class TestBoundsWhatItReads:
    async def test_it_stops_reading_a_body_that_never_ends(
        self,
        fetcher: AiohttpWebContentFetcher,
        serve: Serve,
        treat_local_as_public: None,
    ) -> None:
        # `response.text()` reads until the server stops sending, so a link to a
        # disk image is enough to exhaust memory. This server would happily send
        # far more than the ceiling; the test passing at all is the proof that
        # the fetcher stopped asking.
        sent = 0

        async def handler(request: web.Request) -> web.StreamResponse:
            nonlocal sent
            response = web.StreamResponse(headers={"Content-Type": "text/html"})
            await response.prepare(request)
            block = b"<p>" + b"y" * 60_000 + b"</p>"
            try:
                # Comfortably past the ceiling, and bounded so a fetcher that
                # never stopped fails the test instead of hanging the suite.
                for _ in range(MAX_PAGE_BYTES // len(block) + 20):
                    await response.write(block)
                    sent += len(block)
            except ConnectionResetError:
                pass
            return response

        application = web.Application()
        application.router.add_get("/", handler)
        base = await serve(application)

        text = await fetcher.fetch_all([f"{base}/"])

        assert text
        # The text budget applies on top of the byte budget.
        assert len(text) <= MAX_PAGE_CHARS

    @pytest.mark.parametrize("content_type", ["application/pdf", "video/mp4", "application/zip"])
    async def test_it_refuses_something_that_is_not_a_page(
        self,
        content_type: str,
        fetcher: AiohttpWebContentFetcher,
        serve: Serve,
        treat_local_as_public: None,
    ) -> None:
        # Pointing lxml at a tarball spends the whole byte budget to produce
        # nothing. The refusal is logged and dropped by `fetch_all`, so the
        # observable result is an empty answer rather than an exception.
        application = web.Application()
        application.router.add_get("/", page("not markup", content_type=content_type))
        base = await serve(application)

        assert await fetcher.fetch_all([f"{base}/"]) == ""


class TestFollowsRedirectsSafely:
    async def test_it_follows_a_redirect_and_reads_the_destination(
        self,
        fetcher: AiohttpWebContentFetcher,
        serve: Serve,
        treat_local_as_public: None,
    ) -> None:
        async def redirect(_request: web.Request) -> web.Response:
            raise web.HTTPFound("/destination")

        application = web.Application()
        application.router.add_get("/", redirect)
        application.router.add_get("/destination", page(ARTICLE))
        base = await serve(application)

        assert await fetcher.fetch_all([f"{base}/"]) == "The readable part."

    async def test_it_re_checks_every_hop_rather_than_only_the_first_url(
        self,
        fetcher: AiohttpWebContentFetcher,
        serve: Serve,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The reason redirects are followed by hand. A guard that only inspects
        # the URL the user typed is no guard at all: a public host is free to
        # answer with "Location: http://169.254.169.254/", and aiohttp would
        # follow it straight to the cloud metadata service.
        async def redirect(_request: web.Request) -> web.Response:
            raise web.HTTPFound("http://169.254.169.254/latest/meta-data/")

        application = web.Application()
        application.router.add_get("/", redirect)
        base = await serve(application)

        # Only the loopback the test server is on is treated as public; the
        # metadata address is still refused.
        monkeypatch.setattr(
            "app.adapters.web.aiohttp_scraper.is_blocked_address",
            lambda address: address.startswith("169.254."),
        )

        with pytest.raises(UnsafeUrl, match=re.escape("169.254.169.254")):
            await fetcher._fetch(f"{base}/")

    async def test_a_relative_location_is_resolved_against_the_hop_it_came_from(
        self,
        fetcher: AiohttpWebContentFetcher,
        serve: Serve,
        treat_local_as_public: None,
    ) -> None:
        async def redirect(_request: web.Request) -> web.Response:
            return web.Response(status=302, headers={"Location": "../destination"})

        application = web.Application()
        application.router.add_get("/nested/start", redirect)
        application.router.add_get("/destination", page(ARTICLE))
        base = await serve(application)

        assert await fetcher.fetch_all([f"{base}/nested/start"]) == "The readable part."

    async def test_a_redirect_loop_is_abandoned(
        self,
        fetcher: AiohttpWebContentFetcher,
        serve: Serve,
        treat_local_as_public: None,
    ) -> None:
        async def forever(_request: web.Request) -> web.Response:
            raise web.HTTPFound("/loop")

        application = web.Application()
        application.router.add_get("/loop", forever)
        base = await serve(application)

        with pytest.raises(UnsafeUrl, match=re.escape(f"more than {MAX_REDIRECTS} redirects")):
            await fetcher._fetch(f"{base}/loop")


class TestRefusesPrivateTargets:
    """The SSRF guard, with the real predicate and no patching anywhere."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:6333/collections",
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.5/",
            "http://192.168.1.1/",
            "http://[::1]/",
            # 10.0.0.1 wearing an IPv6 costume. None of the v6 predicates would
            # call this private on their own.
            "http://[::ffff:10.0.0.1]/",
        ],
    )
    async def test_a_literal_private_address_never_leaves_the_process(
        self, url: str, fetcher: AiohttpWebContentFetcher
    ) -> None:
        # Refused by `FetchableUrl.parse` before a socket is opened: everything
        # an attacker gets by naming one of these is infrastructure that trusts
        # anything already inside the network.
        with pytest.raises(UnsafeUrl):
            await fetcher._fetch(url)

    @pytest.mark.parametrize(
        "url", ["file:///etc/passwd", "ftp://example.com/x", "gopher://x/", "/relative"]
    )
    async def test_only_http_and_https_are_fetchable(
        self, url: str, fetcher: AiohttpWebContentFetcher
    ) -> None:
        with pytest.raises(UnsafeUrl):
            await fetcher._fetch(url)

    async def test_a_hostname_that_resolves_to_a_private_address_is_refused(
        self, fetcher: AiohttpWebContentFetcher
    ) -> None:
        # The literal check in the domain cannot catch this one - "localhost"
        # is not an address - so the adapter resolves the name and judges every
        # answer. This is real DNS: `localhost` resolves through the system
        # resolver like any other name.
        with pytest.raises(UnsafeUrl, match="resolves to"):
            await fetcher._fetch("http://localhost:6333/")

    async def test_a_host_that_does_not_resolve_is_refused_rather_than_dialled(
        self, fetcher: AiohttpWebContentFetcher
    ) -> None:
        with pytest.raises(UnsafeUrl, match="does not resolve"):
            await fetcher._fetch("http://no-such-host.invalid/")


class TestFetchAll:
    async def test_no_urls_means_no_requests(self, fetcher: AiohttpWebContentFetcher) -> None:
        assert await fetcher.fetch_all([]) == ""

    async def test_it_joins_what_came_back_from_every_page(
        self,
        fetcher: AiohttpWebContentFetcher,
        serve: Serve,
        treat_local_as_public: None,
    ) -> None:
        application = web.Application()
        application.router.add_get("/one", page("<html><main>First.</main></html>"))
        application.router.add_get("/two", page("<html><main>Second.</main></html>"))
        base = await serve(application)

        assert await fetcher.fetch_all([f"{base}/one", f"{base}/two"]) == "First. Second."

    async def test_one_dead_link_does_not_cost_the_others_their_content(
        self,
        fetcher: AiohttpWebContentFetcher,
        serve: Serve,
        treat_local_as_public: None,
    ) -> None:
        # `return_exceptions=True` and the isinstance check. Without them, one
        # 500 in a message full of links would lose every page in it.
        async def fails(_request: web.Request) -> web.Response:
            raise web.HTTPInternalServerError

        application = web.Application()
        application.router.add_get("/good", page("<html><main>Kept.</main></html>"))
        application.router.add_get("/bad", fails)
        base = await serve(application)

        result = await fetcher.fetch_all([f"{base}/bad", f"{base}/good", "http://127.0.0.1/"])

        assert result == "Kept."

    async def test_every_page_that_failed_leaves_an_empty_answer_rather_than_an_error(
        self, fetcher: AiohttpWebContentFetcher
    ) -> None:
        assert await fetcher.fetch_all(["http://10.0.0.1/", "not-a-url"]) == ""

    async def test_it_sends_a_user_agent_that_identifies_the_client(
        self,
        fetcher: AiohttpWebContentFetcher,
        serve: Serve,
        treat_local_as_public: None,
    ) -> None:
        # aiohttp's default agent is rejected outright by a great many sites -
        # Wikimedia answers it with a flat 403 - which silently cost us the
        # content of every wikipedia.org link.
        seen: list[str] = []

        async def handler(request: web.Request) -> web.Response:
            seen.append(request.headers.get("User-Agent", ""))
            return web.Response(body=ARTICLE.encode(), content_type="text/html")

        application = web.Application()
        application.router.add_get("/", handler)
        base = await serve(application)

        await fetcher.fetch_all([f"{base}/"])

        assert seen and seen[0].startswith("TalkToWeb/")
        assert "aiohttp" in seen[0]
