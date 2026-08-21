# Talk to the Web

RAG chat app: paste a URL or upload a PDF/DOCX/text file, then chat about it
with an LLM agent that can retrieve your documents, fetch a web page, or
search the web. FastAPI (Python 3.13, `uv`) backend, React 19/Vite frontend,
Postgres, Qdrant, Groq (LLM), Gemini (embeddings), Tavily (web search),
Deepgram (live speech-to-text).

The backend is an onion-architecture reference implementation
(domain → application → adapters/api), mechanically enforced by
`lint-imports`.

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

Compose brings up Postgres and Qdrant, waits for both to report healthy, runs
`alembic upgrade head` as a one-shot `migrate` service, then starts the
backend — nothing creates tables at application startup, in any environment.

## Tests

```bash
cd backend && uv run pytest
cd frontend && npm test
```

See [backend/README.md](backend/README.md) and [frontend/README.md](frontend/README.md)
for integration tests, linting, and type-checking commands.
