"""`score_tool_call` and the `allowed` tools a task may also complete."""

from evals.metrics.tools import score_tool_call


def test_an_allowed_extra_call_passes_and_is_not_over_called() -> None:
    """tools-016: the agent searched, then opened a link the search returned.
    The routing rule allows that, so the trial must not fail."""
    score = score_tool_call(
        ["search_web", "fetch_web_pages"], ["search_web"], allowed=["fetch_web_pages"]
    )

    assert score.exact_match
    assert score.over_called == ()
    assert score.precision == 1.0
    assert score.recall == 1.0


def test_an_allowed_call_alone_does_not_satisfy_the_expected_tool() -> None:
    """Allowed is not a substitute: fetching without searching still misses
    the tool the task requires."""
    score = score_tool_call(["fetch_web_pages"], ["search_web"], allowed=["fetch_web_pages"])

    assert not score.exact_match
    assert score.missing == ("search_web",)
    assert score.recall == 0.0


def test_an_extra_call_that_is_not_allowed_still_fails() -> None:
    score = score_tool_call(
        ["search_web", "retrieve_documents"], ["search_web"], allowed=["fetch_web_pages"]
    )

    assert not score.exact_match
    assert score.over_called == ("retrieve_documents",)
    assert score.precision == 0.5
