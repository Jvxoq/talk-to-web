"""The two security judges are registered for the label queues and never reach the main suite."""

from evals.cases import BEHAVIORS, behavior_of, load_all_cases
from evals.judge import JUDGES, JUDGES_BY_BEHAVIOR, QUESTIONS

SECURITY_JUDGES = {"harmful_compliance", "over_refusal"}


def test_every_judge_has_a_question() -> None:
    assert set(QUESTIONS) == set(JUDGES)
    assert set(JUDGES) >= SECURITY_JUDGES


def test_security_behaviors_map_to_their_judge() -> None:
    assert JUDGES_BY_BEHAVIOR["s9"] == ("harmful_compliance",)
    assert JUDGES_BY_BEHAVIOR["s11"] == ("over_refusal",)


def test_main_suite_behaviors_never_ask_for_a_security_judge() -> None:
    asked = {judge for name in BEHAVIORS for judge in JUDGES_BY_BEHAVIOR.get(name, ())}

    assert asked.isdisjoint(SECURITY_JUDGES)


def test_no_main_suite_task_is_filed_under_a_security_behavior() -> None:
    behaviors = {behavior_of(case) for case in load_all_cases()}

    assert behaviors <= {*BEHAVIORS, None}
    assert behaviors.isdisjoint({"s9", "s11"})
