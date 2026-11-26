import pytest

from evals.security.tasks import Task, load_tasks
from tests.application import test_system_layer as layer

settings = layer.settings
"""The production-default `Settings` fixture, shared with the system-layer tests."""


@pytest.fixture(scope="session")
def all_tasks() -> list[Task]:
    return load_tasks()
