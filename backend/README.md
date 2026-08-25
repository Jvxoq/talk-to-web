# Backend

FastAPI (Python 3.13, `uv`) service for Talk to the Web — RAG chat over
uploaded documents, with auth and live speech-to-text. See the
root `ARCHITECTURE.md` for the whole-picture view (product, deployment); this file
is backend internals.

## Commands

```bash
uv sync --all-groups
uv run fastapi dev app/main.py         # dev server on :8000

uv run pytest                          # full suite, no infrastructure needed
uv run pytest tests/domain/test_domain.py::test_name   # one test

# Integration tests, deselected by default. Needs `docker compose up -d postgres qdrant`.
INTEGRATION_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/postgres \
INTEGRATION_QDRANT_URL=http://localhost:6333 \
uv run pytest -m integration

uv run mypy app tests                  # strict; this is what makes the ports real
uv run ruff check app tests
uv run lint-imports                    # fails if a layer imports outward
```

Migrations own the schema in **every** environment — `alembic upgrade head`
is the only thing that creates tables, in local dev too:

```bash
uv run alembic upgrade head
uv run alembic revision --autogenerate -m "what changed"
```

`env.py` connects with `DATABASE_MIGRATION_URL` when set and `DATABASE_URL`
otherwise — managed Postgres publishes a pooled endpoint for the app and a
direct one for DDL. One table set is not Alembic's: LangGraph's
`checkpoint_*` tables are created by `checkpointer.setup()` at startup, on
purpose, and `include_name` in `migrations/env.py` keeps autogenerate from
proposing to drop them.

## Architecture

```
        api ──┐
              ├──▶ application ──▶ domain
   adapters ──┘
```

Imports only point inward, enforced by `lint-imports`
(`[tool.importlinter]` in `pyproject.toml`):

- `app/domain/` is stdlib-only — no FastAPI, SQLAlchemy, pydantic, aiohttp.
- `app/application/` may not import `app.adapters` or any framework. Use
  cases depend on `typing.Protocol` ports declared in
  `application/*/ports.py`; adapters satisfy them structurally, never by
  inheritance.
- `app/api/` may not import `app.adapters`.
- Nothing outside `app/composition.py` may import `app/settings.py`.

`app/composition.py` is the single place allowed to import in every
direction, and the only place a client, engine, or session factory is
constructed. Nothing is built at import time, so importing any module needs
no API keys.

Layers, by directory:

- `app/domain/` — entities, value objects, domain errors, one package per
  bounded context (`chat`, `identity`, `ingestion`, `transcription`, `usage`).
  Pure Python.
- `app/application/` — use cases (one class per operation, collaborators
  injected via `__init__`) and the `Protocol` ports they depend on. `chat/`
  also holds the LangGraph agent graph, nodes, and tools.
- `app/adapters/` — port implementations: `persistence` (SQLAlchemy repos +
  unit of work), `vector` (Qdrant), `embedding` (Gemini), `llm` (LangChain
  chat model + token counter), `web` (aiohttp scraper, Tavily search),
  `extraction` (PDF/DOCX/plain-text), `storage` (local file storage),
  `security` (Argon2 hashing, JWT, refresh tokens, in-memory rate limiter),
  `transcription` (Deepgram), `time`.
- `app/api/` — FastAPI routers, Pydantic schemas, SSE wire format,
  dependency wiring, error mapping.
- `app/observability/` — ambient logging/Sentry/request-id, importable from
  any layer.

Tests mirror the layers: `tests/domain/` (no fixtures, no async, no I/O),
`tests/application/` (use cases against `tests/fakes.py`), `tests/api/`
(real app, stub container). `tests/integration/` is deselected by default
and needs real Postgres/Qdrant — see Commands above.

## Features (endpoints → use cases)

| Endpoint | Use case | What it does |
|---|---|---|
| `POST /generate/text/` | `GenerateReply` | Streams (SSE) an agent reply for a conversation turn; agent may call `retrieve_documents`, `fetch_web_pages`, `search_web` |
| `POST /conversations/` | `StartConversation` | Creates a conversation for the current user |
| `GET /conversations/` | `ListConversations` | Lists the user's conversations |
| `GET /conversations/{id}` | `GetConversation` | Loads one conversation with its messages |
| `POST /conversations/{id}/messages` | `RecordExchange` | Persists a user/assistant message pair |
| `POST /conversations/{id}/delete` | `DeleteConversation` | Deletes a conversation (POST, not DELETE — CORS only allows GET/POST/OPTIONS; see frontend README) |
| `POST /upload/file/` | `UploadDocument` → `IndexDocument` | Extracts text (PDF/DOCX/plain text), chunks, embeds (Gemini), stores in Qdrant + Postgres, and writes a short digest to `documents.summary` |
| `GET /documents/` | `ListDocuments` | Lists the user's indexed documents |
| `POST /documents/{id}/delete` | `DeleteDocument` | Removes a document's chunks from Qdrant and its row from Postgres |
| `POST /auth/register` | `RegisterUser` | Creates an account (Argon2-hashed password) |
| `POST /auth/login` | `AuthenticateUser` | Issues an access token + refresh cookie |
| `POST /auth/refresh` | `RefreshSession` | Rotates the refresh token, issues a new access token |
| `POST /auth/logout` | `RevokeSession` | Revokes the current refresh token |
| `GET /auth/me` | `IdentifyRequest` | Returns the current user from the access token |
| `GET /models/` | — (delivery config) | Lists selectable chat models (`chat_models` setting) |
| `GET /health` | — | Static liveness |
| `GET /ready` | `CheckReadiness` | Probes Postgres + Qdrant concurrently, 503 with a per-dependency `checks` map on failure |
| `WS /ws/transcribe/` | `TranscribeStream` | Streams mic audio to Deepgram, streams partial/final transcripts back |

## Operational concerns

- **Cost drivers.** Every agent lap (main model + tool calls), condenser
  call, embedding call, Tavily search, and Deepgram stream is metered usage.
  Indexing an upload adds one more condenser call, for the document digest.
  `agent_max_tool_iterations` (default 8) bounds a looping agent, and its last
  lap is given no tools, so hitting the ceiling produces a text answer rather
  than a cut-off reply. `agent_history_token_budget` /
  `agent_recent_token_budget` / `agent_tool_output_token_budget` bound what
  gets resent on every lap of a long conversation — the agent replays the
  whole thread each round trip, so an uncompressed history is the main way
  this gets expensive. All three were tripled deliberately: the old numbers
  were tuned for a free tier's rate limit, compressed a thread after two
  exchanges, and handed every retrieval to the condenser before the model had
  read it.
- **`agent_max_request_tokens` is the ceiling that matters.** The three
  budgets above shape what stays in view for answer quality; this one maps to
  what the provider actually rejects, and it is checked on every lap. The tool
  schemas are measured once at startup by `build_agent_graph` and subtracted
  from it, so the setting holds the provider's number rather than a
  pre-shrunk guess.
- **Rate limiting.** Auth endpoints are limited in-memory
  (`auth_rate_limit_attempts` per `auth_rate_limit_window_seconds`) — correct
  only for a single backend process; a multi-instance deployment needs a
  shared store.
- **Readiness vs liveness.** `/health` never touches a dependency, on
  purpose — a liveness probe that does turns a two-second blip into every
  instance restarting at once. `/ready` does the real check, under
  `READINESS_TIMEOUT_SECONDS` (2s default), concurrently across dependencies.
- **Observability.** Every request gets an id (`RequestIdMiddleware`,
  outermost middleware) that appears in the `X-Request-ID` response header,
  every log line, the body of a 500, and the `request_id` Sentry tag. Sentry
  is opt-in via `SENTRY_DSN`; `send_default_pii` stays off deliberately (no
  cookies, no `Authorization`, no request bodies — an event can't carry a
  refresh token or a chat).
- **Housekeeping job.** `refresh_tokens` rows are never deleted on rotation
  or sign-out, only marked `revoked_at` — a row that's gone can't be told
  apart from one that never existed, which is what reuse detection depends
  on. `app/cleanup_expired_refresh_tokens.py` deletes rows past
  `REFRESH_TOKEN_CLEANUP_RETENTION_SECONDS` (30 days default). It's a
  standalone entry point, not a route, wired as a `cleanup-refresh-tokens`
  service behind a `jobs` compose profile — run it from cron or a
  Kubernetes CronJob, it does not run on a plain `up`:
  ```bash
  docker compose --profile jobs run --rm cleanup-refresh-tokens
  ```

## Known limits

- **Uploads are on local disk** (`local_file_storage.py`). Fine for one
  instance; does not survive replacing it or scale past one box. Move to S3
  first.
- **In-memory rate limiter and LangGraph's Postgres checkpointer both
  assume a single process** — horizontal scaling needs a shared rate-limit
  store before it needs anything else here.
- `created_at` / `updated_at` are `DateTime` without timezone, so Postgres
  drops the offset on values the models write as timezone-aware. Worth
  fixing before there's production data to migrate.
- No metrics/OpenTelemetry yet — deliberately deferred, since they need
  somewhere to send data (infrastructure decision, not a code change).
- **The default chat model routes tools poorly.**
  `deepseek-ai/DeepSeek-V4-Flash-0731` measured 0.36 exact-match on
  `--suite tools` (n=11), mostly by reaching for `search_web` on
  document-scoped questions. Kept as the default by explicit choice; the
  routing policy is what makes that survivable. Re-run `evals --suite tools`
  before adding any model to `llm_models`.
- **`agent_max_request_tokens` has not been confirmed against a real 400.**
  The previous value was Groq's measured 8,000-token limit. The current one is
  a provisional read of Together's larger window — tighten it if a 400 appears.
- **The condenser can fail silently.** It is allowed to, on purpose, but the
  cost is not silent: a thread over the history budget that cannot be
  summarized falls through to dropping older messages, so the conversation
  loses its past. `Condenser failed on ...` in the logs is the line that says
  so, and a retired model id is how it happened before.

## Key decisions

- **Tool routing is enforced in `ToolRegistry.invoke`, not in a prompt.** A
  `ToolRoutingPolicy` — tool names supplied by `composition.py`, never known
  to the registry — holds `search_web` back until `retrieve_documents` has run
  on a turn the user framed as being about their own files, and refuses
  `retrieve_documents` outright on an account with nothing indexed. It lives
  at the same choke point as the untrusted-content fence for the same reason:
  a rule enforced in one node, or one tool, is a rule the next caller forgets.
  An empty retrieval opens the search immediately, so the model still chooses
  *whether* to go to the web.
- **`ToolContext` is written by us, never by the model.** `owner_id`,
  `document_scoped`, `has_documents` and `prior_tools` all ride the run config
  or the history, not the tool arguments. `document_scoped` is decided once
  per request by `GenerateReply` rather than re-derived by the tool node,
  because the summarize node can replace the history a re-derivation would
  read — and that failure direction is the silent one, "search allowed".
  `has_documents` defaults to `True`, so an unknown answer costs one wasted
  call instead of telling a user with a full shelf that they have none.
- **The agent is told what the user uploaded.** `IndexDocument` writes a short
  digest per document (the `Condenser` again, satisfying ingestion's own
  `DocumentSummarizer` port structurally), and `GenerateReply` appends the
  newest few to the **user turn** — the system prompt is written once per
  thread, documents arrive between turns. The digest is fenced with the same
  `ToolOutputGuard` tool results get, because it is written from an uploaded
  file and is replayed every turn. An account with none is told so outright,
  in an unfenced bracketed line: silence read as "unknown, try anyway".
- **`Protocol` ports, not ABCs.** Adapters satisfy `application/*/ports.py`
  structurally; `mypy --strict` over `tests/fakes.py` is what proves a fake
  still matches its port, without an inheritance chain forcing the
  application layer to know adapter types exist.
- **A `SqlAlchemyUnitOfWork` per use-case call.** Sessions aren't shareable
  across concurrent requests, so the use case owns its own transaction
  boundary rather than sharing one from a request-scoped dependency.
- **`Depends()` providers do one container lookup, no branching or I/O.** A
  provider that needs a branch, an `await`, or any I/O is a sign the logic
  belongs in a use case instead. The API sees the container through the
  `Container` Protocol in `dependencies.py`, not the concrete
  `AppContainer` — adding a use case means updating the Protocol,
  `AppContainer`, and the `container = AppContainer(...)` wiring.
- **Domain errors never become `HTTPException` inside a use case.** The
  mapping lives in one place, `_STATUS` in `app/api/errors.py`, so a 5xx
  never leaks internals and the mapping is auditable in one file.
- **Streaming stays behind one seam.** `GenerateReply` yields `ReplyEvent`
  DTOs; only `app/api/v1/sse.py` knows the SSE wire format. The WebSocket
  handler implements a `ClientTransport` port so `TranscribeStream` never
  names WebSockets directly — either delivery mechanism could be swapped
  without touching the use case.
- **`CORSMiddleware` does not run on a WebSocket handshake.**
  `/ws/transcribe/` checks `Origin` against `websocket_origins` itself and
  closes with 1008 before `accept()`; a missing `Origin` is refused too.
- **Provider-agnostic LLM config.** `llm_provider` is a plain string
  resolved through LangChain's `init_chat_model` (`together`, `groq`,
  `openai`, `anthropic`, `google_genai`, `ollama`, ...). Model choice matters more
  than it looks: `llm_models` only lists models measured to reliably emit
  well-formed tool calls against this app's tool schemas — several
  candidates were dropped for refusing tools outright or emitting malformed
  calls under load.
- **Access tokens are stateless, refresh tokens are revocable rows.** A
  15-minute JWT needs no database hit to verify; the 14-day refresh token is
  rotated on use and revocable, and a revoked token resurfacing is treated
  as reuse, which kills the whole session family — see
  `app/application/identity/sessions.py` (`RefreshSession`).
- **Migrations own the schema everywhere, including local dev.** The
  composition root no longer calls `Base.metadata.create_all` under any
  `ENVIRONMENT`; `alembic upgrade head` is the only path to a table
  existing, so there's no drift between how a laptop and production got
  their schema.
