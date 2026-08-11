"""`/ready` against the dependencies it exists to report on.

The unit tests in `tests/application/test_readiness.py` prove what
`CheckReadiness` does with a probe's answer. What they cannot prove is that the
probes ask a real Postgres and a real Qdrant a question those servers
understand - and a readiness endpoint that is wrong about that is worse than
none, because it reports healthy while nothing works.
"""

from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.adapters.persistence.readiness import PostgresProbe
from app.adapters.vector.readiness import QdrantProbe
from app.application.health.use_cases.check_readiness import CheckReadiness

# Reserved by IANA as "port 0 is not a port": nothing can be listening, on any
# machine, so this is a refused connection rather than a flaky one.
UNREACHABLE_POSTGRES = "postgresql+psycopg://nobody:nobody@127.0.0.1:1/nowhere"
UNREACHABLE_QDRANT = "http://127.0.0.1:1"


class TestPostgresProbe:
    async def test_it_passes_against_a_live_database(self, engine: AsyncEngine) -> None:
        await PostgresProbe(engine).check()

    async def test_a_refused_connection_is_not_ready(self) -> None:
        """Asserted through the use case rather than on the driver's exception type.

        Which exception psycopg raises for a refused socket is its business and
        changes between versions; that the verdict is "not ready" is ours.
        """
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
        """A brand-new deployment has uploaded nothing yet, and is still ready.

        The collection is created on first upload. A probe that waited for it
        would fail the very first rollout of a new environment, which is the one
        rollout nobody can debug by comparing against a working instance.
        """
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
