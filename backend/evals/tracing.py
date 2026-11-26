"""One LangSmith trace per eval trial, so a person can label it.

Off unless `LANGSMITH_API_KEY` is set. With a key, each trial opens one root
trace. The agent's LangGraph run, its model calls and each tool step's output
nest under it on their own, through LangChain's built-in LangSmith support, so
the trace shows the question, every tool result and the final answer with no
tracing code in `app/`.

The judge runs outside the trial trace (see `evals.__main__`). A person
labeling the trace then cannot see the judge's verdict, which is what keeps
the labels blind (eval-design.md, item 10).

Tracing is switched on per trial with `tracing_context`, not with the global
`LANGSMITH_TRACING` variable. A global switch would also trace every judge
call and every fixture embedding as traces of their own, and those count
against the free plan's 5,000 traces a month.
"""

import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import langsmith
from loguru import logger
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LangSmithSettings(BaseSettings):
    """The two settings eval tracing and labeling read. Both from `.env`."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "talk-to-web-evals"

    def client(self) -> langsmith.Client | None:
        if self.langsmith_api_key is None:
            return None
        return langsmith.Client(api_key=self.langsmith_api_key.get_secret_value())


@dataclass(slots=True)
class TrialTrace:
    """The open trace of one trial. `trace_id` is `None` with tracing off."""

    trace_id: str | None
    outputs: dict[str, Any] = field(default_factory=dict)


class TrialTracer:
    """Opens one root trace per trial, or does nothing without a client."""

    def __init__(self, client: langsmith.Client | None, project: str) -> None:
        self._client = client
        self._project = project

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @asynccontextmanager
    async def trial(
        self, name: str, *, inputs: Mapping[str, Any], metadata: Mapping[str, Any]
    ) -> AsyncIterator[TrialTrace]:
        """Trace everything run inside the block as one trial.

        Fill `outputs` on the yielded handle before the block ends. They become
        the trace's output, which is what the labeling screen shows first.
        """
        if self._client is None:
            yield TrialTrace(trace_id=None)
            return
        with langsmith.tracing_context(
            enabled=True, client=self._client, project_name=self._project
        ):
            async with langsmith.trace(
                name,
                run_type="chain",
                inputs=dict(inputs),
                metadata=dict(metadata),
                client=self._client,
                project_name=self._project,
            ) as run:
                handle = TrialTrace(trace_id=str(run.trace_id))
                try:
                    yield handle
                finally:
                    run.end(outputs=handle.outputs)

    async def flush(self) -> None:
        """Send queued traces before the process exits. Never raises."""
        if self._client is None:
            return
        try:
            await asyncio.to_thread(self._client.flush)
        except Exception as error:  # noqa: BLE001 - a lost trace must not fail the run
            logger.warning(f"Could not flush LangSmith traces: {error}")
