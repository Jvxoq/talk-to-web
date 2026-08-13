# Talk to the Web

A full-stack RAG chat app: paste a URL or upload a PDF/DOCX/text file, then chat
about its content with an LLM agent that can retrieve your documents, fetch a
web page, or search the web. FastAPI (Python 3.13, `uv`) backend, React 19/Vite
frontend, Postgres, Qdrant, Groq (LLM), Gemini (embeddings), Tavily (web
search), Deepgram (live speech-to-text).

The repo also doubles as a reference implementation of onion architecture —
domain, application, adapters/api — mechanically enforced by `lint-imports`.
See `backend/README.md` for that in depth.

## Getting started

```bash
# Backend
cd backend
cp .env.example .env
uv sync --all-groups
uv run alembic upgrade head            # creates the schema
uv run fastapi dev app/main.py

# Frontend (in another terminal)
cd frontend
npm install
npm run dev
```

The app runs on `localhost:5173` and connects to the backend at `localhost:8000`.

Or the whole stack in containers:

```bash
docker compose up --build
```

Compose brings up Postgres and Qdrant, waits for both to report healthy, runs
`alembic upgrade head` as a one-shot `migrate` service, and only then starts
the backend. Nothing creates tables at application startup — in any
environment.

Backend details (architecture, use cases, config) live in
`backend/README.md`; frontend details (feature slices, state, theming) live
in `frontend/README.md`. This file stays at the whole-picture level.

---

## Architecture

```
 ┌────────────┐   HTTP/SSE, WebSocket    ┌─────────────────────────────────┐
 │  frontend   │ ───────────────────────▶│  backend (FastAPI)              │
 │  (React)   │                          │  api ─▶ application ─▶ domain   │
 └────────────┘                          │  adapters ─▶ application/domain │
                                         └───────┬───────────┬─────────────┘
                                                 │           │
                                     Postgres (state)   Qdrant (vectors)
                                                             │
                                                  Groq (LLM) · Gemini (embeddings)
                                         Tavily (web search) · Deepgram (speech-to-text)
```

- **Frontend** talks to the backend over same-origin HTTP/SSE for chat,
  auth and document management, and a direct WebSocket for live
  transcription (the one call that can't go through a same-origin rewrite).
- **Backend** is layered onion-style: `domain` is pure Python with no
  framework imports; `application` holds use cases behind `Protocol` ports;
  `adapters` implement those ports against Postgres, Qdrant, and the LLM/
  search/transcription providers; `api` wires HTTP/WebSocket delivery on top.
  `app/composition.py` is the one place allowed to construct clients and wire
  everything together.
- **Postgres** is the system of record for users, conversations, messages and
  document metadata, plus LangGraph's own checkpoint tables for agent state.
- **Qdrant** holds embedded document chunks for retrieval.

## Features

| Feature | Why it exists | Data flow |
|---|---|---|
| Chat with an LLM agent | Core product loop | Frontend → `POST /generate/text` (SSE) → `GenerateReply` use case → LangGraph agent → tool calls as needed → streamed `ReplyEvent`s |
| Retrieve uploaded documents | Answer questions about a file the user uploaded | Agent tool `retrieve_documents` → Qdrant similarity search → chunks back into the prompt |
| Fetch a web page | Answer questions about a URL the user pastes | Agent tool `fetch_web_pages` → `aiohttp` scraper → page text into the prompt |
| Search the web | Answer questions needing current information | Agent tool `search_web` → Tavily API → results into the prompt |
| Upload PDF/DOCX/text | Bring outside content into the knowledge base | `POST /upload` → extract text → chunk → embed (Gemini) → store in Qdrant, metadata in Postgres |
| Ingest a URL | Same, but source is a web page instead of a file | `POST /ingest-url` → scrape → same extract/chunk/embed pipeline |
| Live voice input | Hands-free question entry | Browser mic → WebSocket `/ws/transcribe/` → Deepgram streaming → partial/final transcripts back to the client |
| Auth (email/password) | Conversations and documents are per-user | Access token (15 min, stateless) + refresh token (14 days, httpOnly cookie, rotated, revocable) |
| Conversation history | Resume a chat later | Persisted in Postgres, loaded on conversation open |
| Long-conversation compression | Keep replies inside the model's token budget | LangGraph agent summarizes older history and condenses large tool outputs via a cheap secondary model |

## Deployment

Four pieces, three of them managed:

```
 Vercel (static)                     EC2
 ┌────────────────┐         ┌──────────────────────┐        Neon Postgres
 │ built frontend │         │ caddy  (TLS, :443)   │───────▶ Qdrant Cloud
 │  vercel.json   │──HTTPS─▶│   └─ backend :8000   │        Groq/Gemini/
 │   rewrites     │         │ migrate (runs once)  │        Deepgram/Tavily
 └────────┬───────┘         └──────────────────────┘
          └──── WebSocket, direct to api.<domain> ────┘
```

Two things about that diagram are load-bearing:

- **HTTP calls go through Vercel's rewrites**, so the browser still sees them
  as same-origin. That is not cosmetic: conversation deletion runs on tab
  close via `navigator.sendBeacon`, beacons cannot send a CORS preflight, and
  a cross-origin JSON POST would need one. Same-origin is what keeps deletes
  working.
- **The WebSocket bypasses Vercel entirely.** Rewrites do not carry an
  Upgrade handshake, so voice input connects to the backend's own domain, set
  through `VITE_WS_URL`. Being genuinely cross-origin, it is also the one
  route the browser will not protect — hence `ALLOWED_WEBSOCKET_ORIGINS`,
  checked by the route itself before the socket is accepted.

### 1. Managed data services

Create a **Neon** project and copy both connection strings — the pooled one
(the host with `-pooler` in it) for the application, and the direct one for
migrations. DDL through a transaction pooler is unreliable, which is why they
are two separate variables.

Create a **Qdrant Cloud** cluster and copy its HTTPS URL and API key.

### 2. Backend on EC2

DNS first: point an A record for `api.<your-domain>` at the instance and open
80 and 443. Caddy requests its certificate on first request, and Let's
Encrypt validates over those ports — without DNS in place beforehand, it will
fail and back off.

```bash
git clone <repo> && cd talk-to-the-web

cp .env.example .env                                # API_DOMAIN=api.your-domain
cp backend/.env.production.example backend/.env.production   # then fill it in

docker compose -f docker-compose.prod.yml up -d --build
curl https://api.<your-domain>/health                # {"status":"ok"}
curl https://api.<your-domain>/ready                 # {"status":"ready","checks":{…}}
```

The two are not the same question. `/health` is liveness and is deliberately
static — a liveness probe that touches the database turns a two-second blip
into every instance restarting at once. `/ready` actually probes Postgres and
Qdrant, concurrently and under `READINESS_TIMEOUT_SECONDS` (2s by default),
and answers **503** with `{"status":"degraded"}` plus a per-dependency
`checks` map when any of them is unreachable. That status code is what a
rollout gate should read; the body names the dependency that was missing, and
nothing more, because the endpoint needs no credentials.

That compose file runs three services and no more: `caddy` (TLS, the only
thing published), `migrate` (`alembic upgrade head`, must exit 0 before
anything serves), and `backend` (no host port — reachable only through
Caddy).

`ENVIRONMENT=production` is set in the compose file rather than the env file,
on purpose: a missing value there would silently fall back to `local` rather
than fail, and it is what switches logging to JSON.

### 3. Frontend on Vercel

Edit `frontend/vercel.json` and replace `api.example.com` in all four
rewrites with your backend domain. Then point Vercel at the `frontend/`
directory and set one environment variable:

```
VITE_WS_URL=wss://api.<your-domain>/ws/transcribe/
```

Leave `VITE_API_URL`, `VITE_UPLOAD_URL`, `VITE_CONVERSATIONS_URL` and
`VITE_MODELS_URL` unset — they default to relative paths, which is exactly
what the rewrites need.

Finally put the Vercel domain in `backend/.env.production` under both
`CORS_ORIGINS` and `ALLOWED_WEBSOCKET_ORIGINS`, and restart the backend.

### Checking it before AWS

`docker-compose.prod.yml` has a `parity` profile that adds the built frontend
behind nginx on `:8080`, so the production images can be exercised locally —
same Dockerfiles, same Caddy TLS setup, on your laptop instead of EC2:

```bash
docker compose -f docker-compose.prod.yml --profile parity up --build
```

With `API_DOMAIN=localhost` Caddy issues a certificate from its own internal
CA instead of calling Let's Encrypt, so `curl -k https://localhost/health`
works on a laptop. Add `http://localhost:8080` to `ALLOWED_WEBSOCKET_ORIGINS`
for the run, since that is the origin the parity frontend is served from.

By default `migrate`/`backend` read `backend/.env.production`, same as the
real deploy — which means an unmodified parity run hits the real Neon `main`
branch and the real Qdrant Cloud cluster. To exercise the managed services
without touching production data, add `docker-compose.parity.yml`:

```bash
cp backend/.env.parity.example backend/.env.parity   # then fill it in
docker compose -f docker-compose.prod.yml -f docker-compose.parity.yml \
  --profile parity up --build
```

Postgres and Qdrant are handled differently, because only one of them has a
safe "same service, different copy" option:

- **Postgres** stays Neon. `backend/.env.parity` sets
  `DATABASE_URL`/`DATABASE_MIGRATION_URL` to a branch created off `main` (a
  full copy-on-write clone, so schema and data look real without ever writing
  back to production). `env_file` lists append across `-f` files rather than
  being replaced, so this file only needs to carry what differs —
  `JWT_SECRET`, provider keys, `CORS_ORIGINS`, etc. still come from
  `.env.production` unchanged.
- **Qdrant** is not Qdrant Cloud at all. It has no branch equivalent, so
  `docker-compose.parity.yml` runs its own throwaway local Qdrant container
  (same as plain local dev) and points `backend`/`migrate` at it via a
  top-level `environment:` override, which wins over whatever `env_file` set
  — that's what actually keeps the real `QDRANT_URL`/`QDRANT_API_KEY` off a
  parity run, not anything in `.env.parity`.

## Operational concerns

- **Cost drivers.** Every model call in the agent loop (main chat model,
  condenser model, embeddings, Tavily search, Deepgram streaming) is metered
  API usage. `agent_max_tool_iterations` bounds a looping agent; the
  condenser's own token budgets (`agent_history_token_budget`,
  `agent_tool_output_token_budget`) bound how much gets resent on every lap
  of a long conversation.
- **Scaling bottleneck: uploads on local disk.** Files land on the
  backend instance's filesystem. Fine for one instance; broken the moment
  there's more than one, or the instance is replaced. Move to S3 before
  scaling past one box.
- **Rate limiting.** Auth endpoints are rate-limited in-memory
  (`auth_rate_limit_attempts` per `auth_rate_limit_window_seconds`), which
  only works correctly with a single backend process — a multi-instance
  deployment needs a shared store (Redis) instead. On top of the per-account
  chat/upload limits sits one deployment-wide ceiling
  (`global_daily_call_budget`, default 200/day) shared by chat replies,
  uploads, URL ingestion and transcription alike — registration has no
  CAPTCHA, so the per-account limits alone only cap one account's spend, not
  the total across as many as someone is willing to create. `Caddyfile` adds a
  second, edge-level layer in front of that: `caddy-ratelimit` (built into the
  `caddy` image by `Caddy.Dockerfile`, since the stock image has no
  rate-limiting module) caps requests per address on `/generate*` and
  `/upload*` before they reach the backend at all. Both are meant to sit behind
  a CDN/WAF (e.g. Cloudflare, proxying DNS in front of the EC2 instance) doing
  the first-line filtering — this layer is the backstop for whatever gets
  through, not a replacement for one.
- **Monitoring.** Every request gets an id (`X-Request-ID`), visible in logs,
  500 bodies, and Sentry events — see `backend/README.md` for how that's
  wired. Sentry itself is opt-in via `SENTRY_DSN`; nothing is sent from local
  runs. Metrics/OpenTelemetry are not built yet — deliberately, since they
  need somewhere to send data, which is an infrastructure decision, not a
  code change.
- **Housekeeping jobs.** Dependabot (weekly, grouped per ecosystem), CodeQL
  (every push/PR to `main` plus weekly), pre-commit hooks mirroring the fast
  half of CI, and a `cleanup-refresh-tokens` job (see `backend/README.md`)
  that must be run on a schedule — it does not run on a plain `docker compose up`.

## Observability

Langfuse is the tracing backend. A `Tracer` port in `app/application/chat/ports.py`
has two adapters: `LangfuseTracer` (when credentials are set) and `NullTracer`
(the default, when unset). Credentials are checked once at startup and fall back
gracefully — no keys means the app runs normally on `NullTracer` and nothing
leaves the process, just like `SENTRY_DSN`.

Spans are nested: `chat.reply` (root) contains `agent.run`, `summarize`, and
`llm.agent` (carries model/tokens/time-to-first-token). Tool calls emit a
`tool.<name>` span with latency and output character count. Tool-output
compression and history summarization get their own `condense.tool_output` and
`condense.summary` spans.

Langfuse is deliberately cloud-hosted, not self-hosted. Self-hosting needs five
extra containers (web, worker, ClickHouse, MinIO, Redis) plus its own Postgres,
and this deployment is 1 GiB where the backend alone is capped at 700 MB. The
hosted free tier costs two environment variables.

Because traces of a chat reply are long and expensive, Langfuse itself is never
on the `/ready` probe — a tracing outage must not return 503. Set `LANGFUSE_*`
keys in `backend/.env.production` to enable it; unset, tracing is off.

[Langfuse trace screenshot goes here]

## Evals and guardrails

**Guardrails** are a domain-layer concern: detectors live in
`app/domain/chat/guardrails.py` as stdlib code that never imports a framework.
Input inspection runs in `GenerateReply` before the stream opens, so refusals
are real 422 responses; redacted text is what the model, the LangGraph checkpointer,
and the trace all see. Tool output is fenced as `<untrusted_content source="...">` by
`ToolRegistry.invoke`, the single choke point — this is indirect prompt-injection
defence and matters because the agent fetches arbitrary URLs and reads user PDFs.
`guardrail_block_on_injection` ships false: flag first, measure the false-positive
rate against real user data, then turn on.

**Evals** measure what matters. The driver is `backend/evals/` — deliberately
outside `app/` — as a CLI runner with JSONL datasets, an LLM judge, and results
in JSON. It runs fast and costs zero API calls on the deterministic red-team
suite (pytest in `tests/application/test_guardrails_redteam.py`), which is gated
in CI.

Run the full suite locally (requires local Qdrant):

```bash
cd backend
docker compose up -d qdrant
uv run python -m evals --suite tools --limit 10 --concurrency 4 --out evals/results/latest.json
uv run python -m evals --suite rag --limit 10 --concurrency 4 --out evals/results/latest.json
```

Sample results from a full run (exact match, overuse, latency, cost per reply):

| Suite | Metric | Value | Notes |
|-------|--------|-------|-------|
| tools | exact-match rate | 1.00 | Agent called the right tool for the task |
| tools | over-calling rate | 0.00 | No wasted tool invocations |
| tools | p50 latency | 8582 ms | Time to final reply |
| tools | mean cost | $0.000447 | Per-reply spend at current pricing |
| rag | hit_rate@3 | 0.33 | Retriever found the answer in top 3 |
| rag | mrr@3 | 0.33 | Mean reciprocal rank of correct chunk |
| rag | mean groundedness | 0.67 | Response factual w.r.t. source |
| rag | mean relevance | 0.67 | Response directly answered the question |

The `rag` suite caught a real miss: asked "who founded Aurora Robotics", the
agent chose `search_web` instead of `retrieve_documents`. That is exactly what
the suite exists for.

## Known limits

- **Uploads are on local disk** — see above. Not shared or durable across
  instances.
- **SSE through Vercel's rewrites** is streamed by Vercel's edge, not
  buffered by a serverless function, but it's worth watching on the first
  long reply. If it ever misbehaves, the fallback is to point
  `VITE_API_URL` straight at `https://api.<domain>/generate/text/` and
  accept CORS for that one call — `VITE_CONVERSATIONS_URL` must stay
  relative regardless, because of the beacon.
- `created_at` / `updated_at` are `DateTime` without timezone, so Postgres
  drops the offset on values the models write as timezone-aware. Worth
  fixing before there is production data to migrate.
- In-memory rate limiting and the LangGraph agent's Postgres checkpointer
  both assume a single backend process; horizontal scaling needs a shared
  rate-limit store first.

## Key decisions

- **Onion architecture with a single composition root**, enforced by
  `lint-imports` rather than convention — a wrong import fails the build
  instead of getting caught in review. See `backend/README.md`.
- **Same-origin proxying for HTTP, direct connection for WebSocket.** Vercel
  can rewrite HTTP but not an `Upgrade` handshake, so the two calls take
  different paths to the same backend. This is what drives both the
  `vercel.json`/`nginx.conf` route lists and `ALLOWED_WEBSOCKET_ORIGINS`.
- **Refresh tokens as revocable rows, access tokens as stateless JWTs.**
  A 15-minute access token needs no database hit to verify; a 14-day refresh
  token is rotated and can be revoked, with reuse-of-a-revoked-token treated
  as a signal to kill the whole session family.
- **Migrations, not `create_all`, own the schema everywhere** — including
  local dev — so there is one source of truth for the schema and no drift
  between how a laptop and production got their tables.
- **Provider-agnostic LLM config.** `llm_provider` is a plain string resolved
  through LangChain's `init_chat_model`, so swapping Groq for another
  provider is an env change, not a code change.
