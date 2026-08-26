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
| Retrieve uploaded documents | Answer questions about a file the user uploaded | Agent tool `retrieve_documents` → Qdrant similarity search, filtered on owner **and** conversation → chunks back into the prompt |
| Fetch a web page | Answer questions about a URL the user pastes | Agent tool `fetch_web_pages` → `aiohttp` scraper → page text into the prompt |
| Search the web | Answer questions needing current information | Agent tool `search_web` → Tavily API → results into the prompt |
| Upload PDF/DOCX/text | Bring outside content into the knowledge base | `POST /upload` (names its conversation) → replace whatever that thread already held → extract text → chunk → embed (Gemini) → store in Qdrant, metadata in Postgres, plus a short digest of the file in `documents.summary` |
| One attachment per conversation | A file the user closed, or replaced, must stop shaping answers | Close button → `POST /documents/{id}/delete`; a second upload removes the first; deleting a thread removes its documents — vectors, stored file and row every time |
| Conversation cap | Each thread carries its own uploads and history, so an account holds few | `MAX_CONVERSATIONS_PER_USER` (2) → `StartConversation` counts in the insert's transaction → 409 past it, never eviction |
| Document digest on every turn | The agent has to know what this thread holds before it can choose a tool | `GenerateReply` reads this conversation's documents once per request → appends `[DOCUMENTS AVAILABLE]` (newest few digests, fenced as untrusted content) or `[NO DOCUMENTS]` to the user turn |
| Documents-before-web routing | A question about the user's own files must not be answered off the web first | `is_document_scoped` (domain, stdlib regex) decides once per request → `ToolRegistry.invoke` refuses `search_web` until `retrieve_documents` has run, and refuses `retrieve_documents` outright on an account with nothing indexed |
| Live voice input | Hands-free question entry | Browser mic → WebSocket `/ws/transcribe/` → Deepgram streaming → partial/final transcripts back to the client |
| Auth (email/password) | Conversations and documents are per-user | Access token (15 min, stateless) + refresh token (14 days, httpOnly cookie, rotated, revocable) |
| Conversation history | Resume a chat later | Persisted in Postgres, loaded on conversation open |
| Long-conversation compression | Keep replies inside the model's token budget | LangGraph agent summarizes older history and condenses large tool outputs via a cheap secondary model |

## Deployment

Four pieces, all four managed:

```
 Vercel (static)                    Render (free web service)
 ┌────────────────┐         ┌──────────────────────────┐     Neon Postgres
 │ built frontend │         │ TLS + proxy (Render's)   │────▶ Qdrant Cloud
 │  vercel.json   │──HTTPS─▶│   └─ backend  :$PORT     │     Together/Gemini/
 │   rewrites     │         │ alembic upgrade on boot  │     Deepgram/Tavily
 └────────┬───────┘         └──────────────────────────┘
          └─── WebSocket, direct to <service>.onrender.com ───┘
```

Two things about that diagram are load-bearing:

- **HTTP calls go through Vercel's rewrites**, so the browser still sees them
  as same-origin. That is not cosmetic: the refresh token is a cookie the
  browser attributes to whichever host answered, and same-origin is what keeps
  it first-party.
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

Put both in the same region, and put the Render service there too. Every chat
turn makes several round trips to each.

### 2. Backend on Render

`render.yaml` at the repo root is the whole deployment. Point Render at this
repo as a Blueprint; it reads that file, builds `backend/Dockerfile`, and
prompts once for every variable marked `sync: false`. Secrets are stored in
Render and never enter the repo or an image layer. `JWT_SECRET` is not
prompted for at all — `generateValue: true` has Render mint it on first deploy
and keep it.

```bash
curl https://<service>.onrender.com/health   # {"status":"ok"}
curl https://<service>.onrender.com/ready    # {"status":"ready","checks":{…}}
```

The two are not the same question. `/health` is liveness and is deliberately
static — a liveness probe that touches the database turns a two-second blip
into a restart. `/ready` actually probes Postgres and Qdrant, concurrently and
under `READINESS_TIMEOUT_SECONDS` (2s by default), and answers **503** with
`{"status":"degraded"}` plus a per-dependency `checks` map when any of them is
unreachable. `healthCheckPath` in `render.yaml` points at `/health`, on
purpose: a Neon blip must not fail a deploy.

Three things the previous EC2 deployment had do not exist here, and each was
doing a job:

- **Caddy is gone.** Render terminates TLS and issues the certificate for the
  service's domain. What went with it is the edge rate limit on `/generate*`
  and `/upload*`, so `global_daily_call_budget` is now the only ceiling
  between a script and the provider bills. Put Cloudflare in front of a custom
  domain if that stops being enough.
- **The one-shot `migrate` service is gone.** Render's pre-deploy hook is
  where `alembic upgrade head` belongs, and it needs a paid instance type. On
  `free` it runs from `dockerCommand` on every boot instead. Alembic is
  idempotent, so a boot with nothing to migrate costs one query.
- **The uploads volume is gone.** A free service has no disk and its
  filesystem is wiped on restart. That is survivable here and only here:
  `LocalFileStorage` writes the file, `PdfTextExtractor` reads it once during
  the same upload request, and after indexing the vectors live in Qdrant and
  the row in Postgres. Nothing opens the file again. Attach a disk before that
  stops being true.

One more property of the free plan is worth planning around: the service sleeps
after roughly 15 minutes without traffic, and the next request pays about a
minute of cold start. `useChat` and the fetch helpers in `lib/http.ts` set no
client-side timeout, so the request waits rather than failing — but the first
visitor after a quiet spell sees a long pause.

### 3. Frontend on Vercel

`frontend/vercel.json` rewrites six paths to the backend. Replace the
`talk-to-web-api.onrender.com` placeholder in all of them with your service's
domain. Then point Vercel at the `frontend/` directory and set one environment
variable:

```
VITE_WS_URL=wss://<service>.onrender.com/ws/transcribe/
```

Leave `VITE_API_URL`, `VITE_UPLOAD_URL`, `VITE_CONVERSATIONS_URL` and
`VITE_MODELS_URL` unset — they default to relative paths, which is exactly
what the rewrites need.

Finally put the Vercel domain into Render under both `CORS_ORIGINS` and
`ALLOWED_WEBSOCKET_ORIGINS`, as a JSON list, and redeploy.

### Checking it before Render

`docker compose up` runs the whole stack locally — its own Postgres and
Qdrant, the `migrate` step, the backend, and the Vite dev server. It does not
build `backend/Dockerfile`, does not run the built frontend, and does not
exercise TLS. TLS is Render's concern now rather than this repo's, but the
image is not: the only thing that builds it is a Render deploy, so a
Dockerfile-level break is found there and nowhere earlier. Build it by hand
before pushing anything that touches dependencies or the Dockerfile:

```bash
docker build -t talk-to-web-backend ./backend
```

The old EC2 setup also had a *parity* profile that served the built frontend
behind nginx. Vercel serves the frontend now, so `frontend/Dockerfile` and
`frontend/nginx.conf` were deleted rather than left as a route list nothing
checks. `frontend/vercel.json` is the only proxy list left.


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
- **Scaling bottleneck: uploads on local disk.** Files land on the backend
  instance's filesystem, which on Render's free plan is wiped on every
  restart. That is survivable only because nothing reads the file after the
  upload request that indexed it. Move to object storage before adding a
  second instance, or before any feature needs the original file back.
- **Rate limiting.** Auth endpoints are rate-limited in-memory
  (`auth_rate_limit_attempts` per `auth_rate_limit_window_seconds`), which
  only works correctly with a single backend process — a multi-instance
  deployment needs a shared store (Redis) instead. On top of the per-account
  chat/upload limits sits one deployment-wide ceiling
  (`global_daily_call_budget`, default 200/day) shared by chat replies,
  uploads and transcription alike — registration has no
  CAPTCHA, so the per-account limits alone only cap one account's spend, not
  the total across as many as someone is willing to create. On EC2 a second,
  edge-level layer sat in front of that, capping requests per address on
  `/generate*` and `/upload*` before they reached the backend at all. Render's
  proxy has no equivalent, so that layer is gone and the daily ceiling is the
  whole defence. It was always meant to sit behind a CDN/WAF (Cloudflare
  proxying DNS in front of a custom domain) doing the first-line filtering, and
  with the edge limit gone that is now the way to get one back.
- **Per-account ceilings.** `MAX_CONVERSATIONS_PER_USER` (default 2) caps how
  many threads an account holds, and a thread holds one document, so the same
  number caps how many files it keeps embedded. Past the cap the API answers
  409 and the sidebar tells the user to delete a thread; the server never
  evicts the oldest one.
- **Monitoring.** Every request gets an id (`X-Request-ID`), visible in logs,
  500 bodies, and Sentry events — see `backend/README.md` for how that's
  wired. Sentry itself is opt-in via `SENTRY_DSN`; nothing is sent from local
  runs. Metrics/OpenTelemetry are not built yet — deliberately, since they
  need somewhere to send data, which is an infrastructure decision, not a
  code change.
- **Housekeeping jobs.** Dependabot (weekly, grouped per ecosystem), CodeQL
  (every push/PR to `main` plus weekly), pre-commit hooks mirroring the fast
  half of CI, and a `cleanup-refresh-tokens` job (see `backend/README.md`)
  that must be run on a schedule. Render cron jobs need a paid plan, so on the
  free one it does not run at all and the `refresh_tokens` table grows slowly;
  `render.yaml` carries the cron service commented out, ready for a paid plan.

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
and this deployment is a 512 MB free instance. The hosted free tier costs two
environment variables.

Because traces of a chat reply are long and expensive, Langfuse itself is never
on the `/ready` probe — a tracing outage must not return 503. Set `LANGFUSE_*`
keys in Render's environment page to enable it; unset, tracing is off.

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

- **Uploads are on local disk, and the disk is ephemeral.** Render's free
  plan gives the service no persistent volume, so the filesystem is wiped on
  every restart and every deploy. Survivable only because nothing reads a
  stored file after the upload request that indexed it. Any feature that needs
  the original file back needs object storage first.
- **SSE through Vercel's rewrites** is streamed by Vercel's edge, not
  buffered by a serverless function, but it's worth watching on the first
  long reply. If it ever misbehaves, the fallback is to point
  `VITE_API_URL` straight at `https://<service>.onrender.com/generate/text/`
  and accept CORS for that one call. `VITE_CONVERSATIONS_URL` must stay
  relative regardless, because `/auth` and the calls that share its origin are
  what keep the refresh cookie first-party.
- `created_at` / `updated_at` are `DateTime` without timezone, so Postgres
  drops the offset on values the models write as timezone-aware. Worth
  fixing before there is production data to migrate.
- In-memory rate limiting and the LangGraph agent's Postgres checkpointer
  both assume a single backend process; horizontal scaling needs a shared
  rate-limit store first.
- **No email verification** — anyone can register with any address they
  like. The global budget cap bounds the damage a throwaway account can do.
- **One attachment per conversation, and no list UI.** A thread holds one
  document, so the attachment chip (upload, then close to delete) is the whole
  document surface in the app. `GET /documents/` is still served and nothing
  in the frontend calls it. A user who wants two files at once needs two
  threads, which the conversation cap then bounds.
- **Qdrant points written before thread scoping match nothing.** Every search
  now filters on owner and `conversation_id`, and older points carry no such
  payload field. That is the intended outcome rather than a gap, but it means
  documents indexed before this change are unreachable and have to be
  re-uploaded.
- **The free service sleeps.** After roughly 15 minutes without traffic
  Render stops the instance, and the next request pays about a minute of cold
  start while the image boots and `alembic upgrade head` runs. No client-side
  timeout is set, so the request waits rather than failing, but the first
  visitor after a quiet spell sees a long pause.
- **No edge rate limit any more.** EC2 had Caddy capping requests per address
  on `/generate*` and `/upload*` before they reached the backend. Render's
  proxy has no equivalent, so `global_daily_call_budget` (200/day) is the only
  ceiling left that a fresh account cannot walk around. Cloudflare in front of
  a custom domain is how to get that layer back.
- **Nothing checks the `vercel.json` route list.** It is the only proxy list
  left, and a backend route missing from it does not error. The request falls
  through to the SPA catch-all and returns `index.html` with a 200. Adding a
  route to the API means adding it there by hand.
- **The housekeeping job does not run.** `cleanup-refresh-tokens` needs a
  scheduler, and Render cron jobs need a paid plan. `render.yaml` carries the
  service commented out; until then the `refresh_tokens` table grows.
- **`agent_max_request_tokens` is provisional.** The old value was Groq's real
  8,000-token limit, confirmed against a 400 in production. The current one is
  a guess at Together's much larger window and has not been confirmed the same
  way. Tighten it if a 400 shows up.

## Key decisions

- **A free managed platform over a self-managed box.** The backend ran on EC2
  behind Caddy, on AWS credits that expire. Render's free web service hosts
  the same `backend/Dockerfile` at no cost and with no expiry, and it
  terminates TLS itself. The price is three real capabilities: the edge rate
  limit, the persistent uploads disk, and the scheduled cleanup job, none of
  which the free plan offers. Each was checked against what this app actually
  needs before it was given up, and each is listed under Known limits rather
  than quietly dropped. The alternative was paying about $10 a month for an
  instance, which is not worth it for a portfolio deployment.
- **Onion architecture with a single composition root**, enforced by
  `lint-imports` rather than convention — a wrong import fails the build
  instead of getting caught in review. See `backend/README.md`.
- **Same-origin proxying for HTTP, direct connection for WebSocket.** Vercel
  can rewrite HTTP but not an `Upgrade` handshake, so the two calls take
  different paths to the same backend. This is what drives both the
  `vercel.json` route list and `ALLOWED_WEBSOCKET_ORIGINS`.
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
- **A document is scoped to its conversation, not to the account.** Every
  Qdrant search filters on owner **and** thread. The owner filter is the
  security boundary; the thread filter is what stops a question in one chat
  being answered out of a file attached to another. It also bounds cost: one
  attachment per thread, and `MAX_CONVERSATIONS_PER_USER` threads per account,
  is how many files one person can keep embedded.
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
