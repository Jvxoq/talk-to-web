"""Tests for `IngestUrl` against fakes - no network, no storage."""

import pytest

from app.application.ingestion.dto import IngestUrlInput
from app.application.ingestion.use_cases.ingest_url import IngestUrl
from app.domain.chat.errors import UnsafeUrl
from app.domain.usage.errors import RateLimited
from tests.fakes import FakeFileStorage, FakeRateLimiter, FakeUrlContentFetcher, UnitOfWorkSpy

OWNER = 7
MAX_BYTES = 10_000_000


def _use_case(
    fetcher: FakeUrlContentFetcher | None = None,
    storage: FakeFileStorage | None = None,
    limiter: FakeRateLimiter | None = None,
) -> tuple[IngestUrl, FakeUrlContentFetcher, FakeFileStorage, FakeRateLimiter]:
    fetcher = fetcher or FakeUrlContentFetcher(text="the full page text")
    storage = storage or FakeFileStorage()
    limiter = limiter or FakeRateLimiter()
    use_case = IngestUrl(fetcher, storage, limiter, MAX_BYTES, uow_factory=UnitOfWorkSpy())
    return use_case, fetcher, storage, limiter


class TestIngestUrl:
    async def test_fetches_and_stores_the_page_as_txt(self) -> None:
        use_case, fetcher, storage, _ = _use_case(
            fetcher=FakeUrlContentFetcher(text="hello from the web")
        )

        result = await use_case(IngestUrlInput(url="https://example.com/article", owner_id=OWNER))

        assert fetcher.calls == ["https://example.com/article"]
        assert result.name.endswith(".txt")
        assert storage.owners == [OWNER]
        [(stored_name, body)] = storage.saved
        assert stored_name == result.name
        assert body == b"hello from the web"
        assert result.reference == f"uploads/{OWNER}/{result.name}"

    async def test_synthetic_name_starts_with_the_host(self) -> None:
        use_case, _, _, _ = _use_case()

        result = await use_case(IngestUrlInput(url="https://example.com/a/b", owner_id=OWNER))

        assert result.name.startswith("example.com")

    async def test_rate_limited_before_any_fetch(self) -> None:
        limiter = FakeRateLimiter(max_attempts=0)
        use_case, fetcher, storage, _ = _use_case(limiter=limiter)

        with pytest.raises(RateLimited):
            await use_case(IngestUrlInput(url="https://example.com", owner_id=OWNER))

        assert fetcher.calls == []
        assert storage.saved == []

    async def test_rejects_a_url_with_no_http_scheme(self) -> None:
        use_case, fetcher, storage, _ = _use_case()

        with pytest.raises(UnsafeUrl):
            await use_case(IngestUrlInput(url="ftp://example.com/file", owner_id=OWNER))

        assert fetcher.calls == []
        assert storage.saved == []

    async def test_rejects_a_literal_private_address(self) -> None:
        use_case, fetcher, storage, _ = _use_case()

        with pytest.raises(UnsafeUrl):
            await use_case(IngestUrlInput(url="http://127.0.0.1/secret", owner_id=OWNER))

        assert fetcher.calls == []
        assert storage.saved == []

    async def test_propagates_a_fetch_failure(self) -> None:
        use_case, _, storage, _ = _use_case(
            fetcher=FakeUrlContentFetcher(fail_with=UnsafeUrl("https://example.com", "blocked"))
        )

        with pytest.raises(UnsafeUrl):
            await use_case(IngestUrlInput(url="https://example.com", owner_id=OWNER))

        assert storage.saved == []
