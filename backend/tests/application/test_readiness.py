"""`CheckReadiness` against fake probes: verdicts, isolation, timeouts."""

import asyncio

from app.application.health.use_cases.check_readiness import CheckReadiness
from tests.fakes import FakeReadinessProbe


async def test_reports_ready_when_every_probe_answers() -> None:
    check = CheckReadiness(
        probes=[FakeReadinessProbe("postgres"), FakeReadinessProbe("qdrant")],
        timeout_seconds=1.0,
    )

    report = await check()

    assert report.ready
    assert {component.name: component.ready for component in report.components} == {
        "postgres": True,
        "qdrant": True,
    }


async def test_one_failure_fails_the_whole_check() -> None:
    """And the healthy component is still reported as healthy.

    The point of the per-component detail: a failed rollout should name the
    dependency that was missing rather than only that something was.
    """
    check = CheckReadiness(
        probes=[
            FakeReadinessProbe("postgres"),
            FakeReadinessProbe("qdrant", error=RuntimeError("connection refused")),
        ],
        timeout_seconds=1.0,
    )

    report = await check()

    assert not report.ready
    assert {component.name: component.ready for component in report.components} == {
        "postgres": True,
        "qdrant": False,
    }


async def test_a_hanging_probe_fails_instead_of_hanging() -> None:
    """The property the endpoint exists for.

    An orchestrator that never gets an answer keeps routing traffic to a process
    that cannot serve it, so a probe that outruns its budget has to count as a
    failure rather than as a wait.
    """
    check = CheckReadiness(probes=[FakeReadinessProbe("qdrant", delay=10.0)], timeout_seconds=0.01)

    report = await check()

    assert not report.ready


async def test_probes_run_concurrently() -> None:
    """Two slow probes must cost one wait, not two.

    Sequential probing is the failure that only shows up in production: each
    dependency is inside the timeout on its own, and the endpoint as a whole is
    outside the orchestrator's.
    """
    probes = [FakeReadinessProbe(f"slow-{index}", delay=0.05) for index in range(4)]
    check = CheckReadiness(probes=probes, timeout_seconds=1.0)

    started = asyncio.get_running_loop().time()
    report = await check()
    elapsed = asyncio.get_running_loop().time() - started

    assert report.ready
    # Generous: the assertion is "not four times 0.05", not a benchmark.
    assert elapsed < 0.15


async def test_no_probes_is_ready() -> None:
    """A deployment with nothing to probe is ready, not indeterminate."""
    assert (await CheckReadiness(probes=[], timeout_seconds=1.0)()).ready
