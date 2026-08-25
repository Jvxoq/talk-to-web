# Talk to the Web

A full-stack RAG chat app: upload a PDF/DOCX/text file, then chat about its
content with an LLM agent that can retrieve your documents, fetch a web page,
or search the web. FastAPI (Python 3.13, `uv`) backend, React 19/Vite
frontend, Postgres, Qdrant, Together (LLM), Gemini (embeddings), Tavily (web
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
                                              Together (LLM) · Gemini (embeddings)
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
| Upload PDF/DOCX/text | Bring outside content into the knowledge base | `POST /upload` → extract text → chunk → embed (Gemini) → store in Qdrant, metadata in Postgres, plus a short digest of the file in `documents.summary` |
| Document digest on every turn | The agent has to know what this account uploaded before it can choose a tool | `GenerateReply` reads the owner's documents once per request → appends `[DOCUMENTS AVAILABLE]` (newest few digests, fenced as untrusted content) or `[NO DOCUMENTS]` to the user turn |
| Documents-before-web routing | A question about the user's own files must not be answered off the web first | `is_document_scoped` (domain, stdlib regex) decides once per request → `ToolRegistry.invoke` refuses `search_web` until `retrieve_documents` has run, and refuses `retrieve_documents` outright on an account with nothing indexed |
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
 │  vercel.json   │──HTTPS─▶│   └─ backend :8000   │        Together/Gemini/
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
git clone <repo> && cd talk-to-web

cp .env.example .env                                # API_DOMAIN=api.your-domain
cp backend/.env.production.example backend/.env.production   # then fill it in

./scripts/check-env.sh                                # fails if a placeholder is still in there
docker compose -f docker-compose.prod.yml up -d --build
curl https://api.<your-domain>/health                # {"status":"ok"}
curl https://api.<your-domain>/ready                 # {"status":"ready","checks":{…}}
```

`scripts/check-env.sh` only checks that the checked-in placeholder strings were
changed, not that what replaced them is correct — it exists to catch exactly
the failure mode of copying the example and forgetting a line, not to
validate credentials.

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

Edit `frontend/vercel.json` and replace `api.example.com` in all five
rewrites with your backend domain. Then point Vercel at the `frontend/`
directory and set one environment variable:

This repo's `frontend/vercel.json` (and the `VITE_WS_URL` example in
`frontend/.env.example`) already point at the live domain,
`talk-to-web.duckdns.org` — this DuckDNS name matches the repo/folder name.
Don't repoint it without also moving DNS and the TLS certificate first; Caddy
holds the certificate for whatever domain is live in production, and changing
it here alone breaks the deployment.

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
branch and the real Qdrant Cloud cluster/collection. To exercise the managed
services without touching production data, add `docker-compose.parity.yml`:

```bash
cp backend/.env.parity.example backend/.env.parity   # then fill it in
docker compose -f docker-compose.prod.yml -f docker-compose.parity.yml \
  --profile parity up --build
```

Postgres and Qdrant both get a safe "same service, different copy" swap,
carried entirely by `backend/.env.parity` — `docker-compose.parity.yml` adds
no service of its own and no `environment:` override, just a later `env_file`
entry that wins for any key it sets:

- **Postgres** points at a branch created off `main` (Neon MCP's
  `create_branch`, or the console) — a full copy-on-write clone, so schema
  and data look real without ever writing back to production.
- **Qdrant** points at the *same* Cloud cluster and API key as production,
  but a separate collection (`knowledge_base_dev` rather than
  `knowledge_base`). Qdrant has no branch equivalent, so a different
  collection on the same cluster is what stands in for one — the app creates
  it on first ingest, nothing to pre-provision.
- `env_file` lists append across `-f` files rather than being replaced, so
  `backend/.env.parity` only needs to carry what differs — `JWT_SECRET`,
  provider keys, `ENVIRONMENT`, etc. still come from `.env.production`
  unchanged. `CORS_ORIGINS`/`ALLOWED_WEBSOCKET_ORIGINS` are the one exception
  worth calling out: `.env.production` names the real Vercel domain, which
  this box never serves during a parity run (the parity frontend is nginx on
  `:8080` instead), so `.env.parity` overrides both to `http://localhost:8080`.

## Operational concerns

- **Cost drivers.** Every model call in the agent loop (main chat model,
  condenser model, embeddings, Tavily search, Deepgram streaming) is metered
  API usage. `agent_max_tool_iterations` bounds a looping agent; the
  condenser's own token budgets (`agent_history_token_budget`,
  `agent_tool_output_token_budget`) bound how much gets resent on every lap
  of a long conversation. Those budgets were tripled to buy answer quality,
  so a long conversation now costs more per lap than it used to — the old
  numbers were tuned for a free tier's rate limit and compressed the thread
  so early that the model forgot what was said. Separately,
  `agent_max_request_tokens` is the ceiling that maps to what the provider
  actually rejects; it is checked on every lap, not only when the thread
  crosses the history budget. Indexing an upload now also costs one condenser
  call, for the document digest.
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
  uploads and transcription alike — registration has no
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

A subset runs on every pull request that touches a prompt, a tool, a guardrail
or the datasets themselves — `.github/workflows/evals.yml`, against a Qdrant
service container and the real model. It selects `--tag ci`, which is every
case that is deterministic and needs nothing but the fixtures; the two cases
that reach the live internet stay out of it, because they fail for the
internet's reasons rather than this repository's. The run exits non-zero when a
case crashes, calls the wrong tool, misses its expected source, or drops a
`must_contain` needle. The judge's scores are reported but never gate a merge —
a 0-1 score from a model is too noisy to block on, while "the reply stopped
saying 2019" is not.

Sample results from a full run (exact match, overuse, latency, cost per reply).
These were measured on Groq, before the switch to Together, so treat them as
the shape of the report rather than current numbers. The Together default
(`deepseek-ai/DeepSeek-V4-Flash-0731`) measured 0.36 exact-match on
`--suite tools` (n=11) and is kept anyway as a known, accepted gap — it is the
gap the routing gate below exists to hold:

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

The prompt was tightened in response, and then backed by something that does
not depend on the model reading it: when the question is phrased as being about
the user's own files, `ToolRegistry.invoke` refuses a `search_web` call until
`retrieve_documents` has run on that turn, and hands the model a sentence
telling it so. An empty retrieval opens the search immediately, so the agent
still decides *whether* to go to the web — only the order is fixed. Cases
`tools-009` through `tools-011` measure it, and the tool-selection metric that
moves is `over-calling rate`.

That gate is one half of a pair. The other half is the **document digest**: on
every turn `GenerateReply` tells the model, in a bracketed
`[DOCUMENTS AVAILABLE]` / `[NO DOCUMENTS]` tag the system prompt names, what
this account has actually uploaded. The digest informs the choice, the gate
makes it binding. `[NO DOCUMENTS]` matters as much as the other branch:
silence read as "unknown, try anyway", so an account with an empty collection
paid for an embedding request and a Qdrant round trip on every turn.

The eval driver itself was hardened at the same time, because a run that
misreports is worse than no run. A case that errored (a rate limit, a dropped
connection) is now dropped from every rate instead of being scored as "called
nothing, said nothing"; fixtures are indexed for *every* suite, since an
unindexed corpus makes `retrieve_documents` come back empty, which is exactly
the state that releases `search_web` — the suite proving the gate works was
measuring a gate with nothing behind it.

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
- **No email verification** — anyone can register with any address they
  like. The global budget cap bounds the damage a throwaway account can do.
- **Documents have no UI any more.** `GET /documents/` and
  `POST /documents/{id}/delete` are still served, but the panel that called
  them is gone, so a user can upload a file and never list or remove it.
  Uploading is now the whole document surface in the app.
- **`agent_max_request_tokens` is provisional.** The old value was Groq's real
  8,000-token limit, confirmed against a 400 in production. The current one is
  a guess at Together's much larger window and has not been confirmed the same
  way. Tighten it if a 400 shows up.

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
  through LangChain's `init_chat_model`, so swapping one provider for another
  is an env change, not a code change. The move from Groq to Together was
  exactly that: two variables, no code touched.
- **Tool routing is enforced, not prompted.** The system prompt asks the model
  to try `retrieve_documents` before `search_web` on a question about the
  user's own files; `ToolRegistry.invoke` makes it binding. A rule that only
  lives in a prompt is a rule the next model ignores, and the measured
  exact-match rate says that happens.
- **The last agent lap gets no tools bound.** Hitting
  `agent_max_tool_iterations` used to end the reply on whatever the model
  streamed alongside its rejected tool call, which was usually nothing.
  Withholding the tools forces a text-only answer instead, so the ceiling is a
  latency and cost knob rather than a cliff a user's reply falls off.
