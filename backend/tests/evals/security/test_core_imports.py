"""The core of `evals.security` must work for any agent, so it must not import this one.

The check is made two ways: the imports in the source, parsed with `ast`, and the
modules that really load when the core is imported in a fresh interpreter.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import evals.security

PACKAGE = Path(evals.security.__file__).resolve().parent
CORE = ("tasks", "graders", "metrics", "gates", "report")
BACKEND = PACKAGE.parents[1]


def imported_modules(source: str) -> set[str]:
    """Every module name a source file imports, with `from x import y` as `x.y` too."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            found.add(base)
            found.update(f"{base}.{alias.name}" for alias in node.names)
        elif (
            isinstance(node, ast.Call)
            and _call_name(node) in {"import_module", "__import__"}
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            found.add(node.args[0].value)
    return found


def _call_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else ""


def forbidden(modules: set[str]) -> list[str]:
    """Imports of the agent (`app`) or of any other `evals` module outside `evals.security`."""
    bad: list[str] = []
    for name in modules:
        in_agent = name == "app" or name.startswith("app.")
        in_other_evals = name.startswith("evals.") and not name.startswith("evals.security")
        if in_agent or in_other_evals:
            bad.append(name)
    return sorted(bad)


@pytest.mark.parametrize("name", CORE)
def test_core_module_imports_nothing_of_the_agent(name: str) -> None:
    source = (PACKAGE / f"{name}.py").read_text(encoding="utf-8")

    assert forbidden(imported_modules(source)) == []


def test_every_core_module_exists() -> None:
    assert [name for name in CORE if not (PACKAGE / f"{name}.py").is_file()] == []


@pytest.mark.parametrize(
    "snippet",
    [
        "import app.settings",
        "from app.application.chat import models",
        "from evals.harness import EvalHarness",
        "from evals import harness",
        "import evals.judge as j",
        "import importlib\nimportlib.import_module('app.settings')",
    ],
    ids=["import app", "from app", "from harness", "from evals import", "alias", "dynamic"],
)
def test_the_guard_flags_a_bad_import(snippet: str) -> None:
    assert forbidden(imported_modules(snippet)) != []


@pytest.mark.parametrize(
    "snippet",
    ["import json", "from evals.security.tasks import Task", "from evals.security import gates"],
)
def test_the_guard_allows_the_package_itself(snippet: str) -> None:
    assert forbidden(imported_modules(snippet)) == []


def test_importing_the_core_loads_no_agent_module() -> None:
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(BACKEND)!r})\n"
        + "".join(f"import evals.security.{name}\n" for name in CORE)
        + "loaded = sorted(m for m in sys.modules if m == 'app' or m.startswith('app.') "
        "or m in ('evals.harness', 'evals.judge', 'evals.cases', 'evals.labels'))\n"
        "print(loaded)\n"
    )

    result = subprocess.run(  # noqa: S603 - this interpreter, a fixed snippet
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True, timeout=25
    )

    assert result.stdout.strip() == "[]"
