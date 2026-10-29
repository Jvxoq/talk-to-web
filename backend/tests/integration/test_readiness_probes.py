"""The readiness probes against real Postgres and real Qdrant."""

from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.adapters.persistence.readiness import PostgresProbe
from app.adapters.vector.readiness import QdrantProbe
from app.application.health.use_cases.check_readiness import CheckReadiness

# Nothing listens on port 1, so the connection is refused, not flaky.
UNREACHABLE_POSTGRES = "postgresql+psycopg://nobody:nobody@127.0.0.1:1/nowhere"
UNREACHABLE_QDRANT = "http://127.0.0.1:1"


class TestPostgresProbe:
    async def test_it_passes_against_a_live_database(self, engine: AsyncEngine) -> None:
        await PostgresProbe(engine).check()

    async def test_a_refused_connection_is_not_ready(self) -> None:
        """Checked via the use case's verdict, not psycopg's exception type."""
        engine = create_async_engine(UNREACHABLE_POSTGRES)
        try:
            report = await CheckReadiness([PostgresProbe(engine)], timeout_seconds=5.0)()
        finally:
            await engine.dispose()

        assert not report.ready


class TestQdrantProbe:
    async def test_it_passes_against_a_live_server_with_no_collections(
        self, qdrant: AsyncQdrantClient
    ) -> None:
        """A new deployment with no collection yet is still ready."""
        await QdrantProbe(qdrant).check()

    async def test_a_refused_connection_is_not_ready(self) -> None:
        client = AsyncQdrantClient(url=UNREACHABLE_QDRANT, check_compatibility=False, timeout=1)
        try:
            report = await CheckReadiness([QdrantProbe(client)], timeout_seconds=5.0)()
        finally:
            await client.close()

        assert not report.ready


class TestTheWholeCheck:
    async def test_both_dependencies_report_ready(
        self, engine: AsyncEngine, qdrant: AsyncQdrantClient
    ) -> None:
        check = CheckReadiness(
            probes=[PostgresProbe(engine), QdrantProbe(qdrant)], timeout_seconds=5.0
        )

        report = await check()

        assert report.ready
        assert {component.name for component in report.components} == {"postgres", "qdrant"}

    async def test_one_dead_dependency_is_named(self, engine: AsyncEngine) -> None:
        """What an operator reads off a failed rollout."""
        dead = AsyncQdrantClient(url=UNREACHABLE_QDRANT, check_compatibility=False, timeout=1)
        check = CheckReadiness(
            probes=[PostgresProbe(engine), QdrantProbe(dead)], timeout_seconds=5.0
        )
        try:
            report = await check()
        finally:
            await dead.close()

        assert not report.ready
        assert {component.name: component.ready for component in report.components} == {
            "postgres": True,
            "qdrant": False,
        }
