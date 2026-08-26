# Talk to the Web

> RAG chat with a production backend — not a notebook demo.

[![Backend CI](https://github.com/Jvxoq/talk-to-web/actions/workflows/backend.yml/badge.svg)](https://github.com/Jvxoq/talk-to-web/actions/workflows/backend.yml)
[![Frontend CI](https://github.com/Jvxoq/talk-to-web/actions/workflows/frontend.yml/badge.svg)](https://github.com/Jvxoq/talk-to-web/actions/workflows/frontend.yml)
[![Evals](https://github.com/Jvxoq/talk-to-web/actions/workflows/evals.yml/badge.svg)](https://github.com/Jvxoq/talk-to-web/actions/workflows/evals.yml)

**[Live demo](https://talk-to-web.vercel.app)** · **[Architecture](ARCHITECTURE.md)** · **[Backend internals](backend/README.md)**

Upload a PDF/DOCX/text file, then chat about it. The agent can retrieve your documents (Qdrant), fetch a web page, or search the web (Tavily) — plus live speech-to-text (Deepgram).

### Why this shows backend + agent skills

This is an **onion-architecture reference** (`domain → application → adapters → api`), mechanically enforced by `lint-imports` — a wrong import fails CI, not code review.

| Layer | What lives there | Rule |
|---|---|---|
| `domain` | Entities, value objects, domain errors | Pure Python, no framework imports |
| `application` | Use cases + `Protocol` ports | Depends only on domain, one class per operation |
| `adapters` | Postgres (SQLAlchemy + UoW), Qdrant, LLM, embeddings, security (Argon2 + JWT) | Implements ports |
| `api` | FastAPI routers, Pydantic schemas, SSE/WebSocket | Wires via `composition.py` — the single place that builds clients |

**Backend highlights:** async FastAPI (Python 3.13, `uv`), PostgreSQL + Alembic (migrations own the schema everywhere, no `create_all`), JWT access (15 min) + refresh (14 days, httpOnly, rotated, revocable), SSE streaming via one seam (`GenerateReply → ReplyEvent → sse.py`), Postgres + Qdrant health checks, Argon2 hashing, in-memory rate limiting, LangGraph agent with Postgres checkpointer.

**Agent highlights:** LangGraph graph with tool routing enforced in `ToolRegistry.invoke` (not just a prompt), Qdrant retrieval filtered on owner + conversation, document digest on every turn.

- Architecture, deployment, features, operational concerns → [ARCHITECTURE.md](ARCHITECTURE.md)
- Backend internals (layers, use cases, config) → [backend/README.md](backend/README.md)
- Frontend internals (feature slices, state, theming) → [frontend/README.md](frontend/README.md)

## Run it locally

```bash
# Backend
cd backend
cp .env.example .env
uv sync --all-groups
uv run alembic upgrade head            # creates the schema
uv run fastapi dev app/main.py         # :8000

# Frontend (in another terminal)
cd frontend
npm install
npm run dev                            # :5173
```

Or the whole stack in containers:

```bash
docker compose up --build
```

Compose brings up Postgres and Qdrant, waits for both to report healthy, runs `alembic upgrade head` as a one-shot `migrate` service, then starts the backend — nothing creates tables at application startup, in any environment.

## Tests

```bash
cd backend && uv run pytest
cd frontend && npm test
```

See [backend/README.md](backend/README.md) and [frontend/README.md](frontend/README.md) for integration tests, linting (`ruff`, `mypy --strict`), and type-checking commands.
