"""Tests for `IngestUrl` against fakes - no network, no storage."""

import pytest

from app.application.ingestion.dto import IngestUrlInput
from app.application.ingestion.use_cases.ingest_url import IngestUrl
from app.domain.chat.errors import UnsafeUrl
from app.domain.usage.errors import RateLimited
from tests.fakes import FakeRateLimiter, FakeUrlContentFetcher, UnitOfWorkSpy

OWNER = 7


def _use_case(
    fetcher: FakeUrlContentFetcher | None = None,
    limiter: FakeRateLimiter | None = None,
) -> tuple[IngestUrl, FakeUrlContentFetcher, FakeRateLimiter]:
    fetcher = fetcher or FakeUrlContentFetcher(text="the full page text")
    limiter = limiter or FakeRateLimiter()
    use_case = IngestUrl(
        fetcher,
        limiter,
        uow_factory=UnitOfWorkSpy(),
        daily_budget=FakeRateLimiter(),
    )
    return use_case, fetcher, limiter


class TestIngestUrl:
    async def test_fetches_and_returns_the_page_text(self) -> None:
        use_case, fetcher, _ = _use_case(fetcher=FakeUrlContentFetcher(text="hello from the web"))

        result = await use_case(IngestUrlInput(url="https://example.com/article", owner_id=OWNER))

        assert fetcher.calls == ["https://example.com/article"]
        assert result.name.endswith(".txt")
        assert result.text == "hello from the web"
        assert result.reference == "https://example.com/article"

    async def test_synthetic_name_starts_with_the_host(self) -> None:
        use_case, _, _ = _use_case()

        result = await use_case(IngestUrlInput(url="https://example.com/a/b", owner_id=OWNER))

        assert result.name.startswith("example.com")

    async def test_rate_limited_before_any_fetch(self) -> None:
        limiter = FakeRateLimiter(max_attempts=0)
        use_case, fetcher, _ = _use_case(limiter=limiter)

        with pytest.raises(RateLimited):
            await use_case(IngestUrlInput(url="https://example.com", owner_id=OWNER))

        assert fetcher.calls == []

    async def test_rejects_a_url_with_no_http_scheme(self) -> None:
        use_case, fetcher, _ = _use_case()

        with pytest.raises(UnsafeUrl):
            await use_case(IngestUrlInput(url="ftp://example.com/file", owner_id=OWNER))

        assert fetcher.calls == []

    async def test_rejects_a_literal_private_address(self) -> None:
        use_case, fetcher, _ = _use_case()

        with pytest.raises(UnsafeUrl):
            await use_case(IngestUrlInput(url="http://127.0.0.1/secret", owner_id=OWNER))

        assert fetcher.calls == []

    async def test_propagates_a_fetch_failure(self) -> None:
        use_case, _, _ = _use_case(
            fetcher=FakeUrlContentFetcher(fail_with=UnsafeUrl("https://example.com", "blocked"))
        )

        with pytest.raises(UnsafeUrl):
            await use_case(IngestUrlInput(url="https://example.com", owner_id=OWNER))

    async def test_a_spent_daily_budget_refuses_before_the_per_user_limit_is_touched(
        self,
    ) -> None:
        fetcher = FakeUrlContentFetcher(text="hello")
        limiter = FakeRateLimiter()
        use_case = IngestUrl(
            fetcher,
            limiter,
            uow_factory=UnitOfWorkSpy(),
            daily_budget=FakeRateLimiter(max_attempts=0),
        )

        with pytest.raises(RateLimited):
            await use_case(IngestUrlInput(url="https://example.com", owner_id=OWNER))

        assert fetcher.calls == []
        assert limiter.hits == {}, "the per-user limit must not be spent on a refused request"
