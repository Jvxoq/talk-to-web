# Talk to the Web: Spec

This file is the whole project on one page. It says what the app is for, what
it does, and why it is built the way it is. It stays at the high level. The
details live in the files listed at the end.

How to use it:

- This file holds the **intent**. The code holds the **behaviour**.
- When the two disagree, one of them has a bug. Decide which one, then fix it.
- Change this file when a goal, a rule or a design choice changes. Do not
  change it for an internal refactor.

---

## 1. Intent

**What it is.** A chat app. You upload a file, then ask questions about it. The
agent can also fetch a web page or search the web. You can speak your question
instead of typing it.

**Why it exists.** It is a portfolio project. It shows a production backend and
a real agent, not a notebook demo. A reviewer should be able to open the live
demo, try it, then read the code and see production habits.

**What it should prove:**

1. **Onion architecture that a machine checks.** A wrong import fails the
   build. It does not wait for a code review.
2. **An agent you can trust.** Rules about tools live in code, not only in a
   prompt. Untrusted text is fenced. Behaviour is measured with evals.
3. **Honest operations.** Every limit is written down. Nothing fails in
   silence. Nothing costs money without a ceiling.

**Who uses it.** A reviewer or a hiring manager who tries the app for a few
minutes. It is not built for open public traffic.

**Non-goals.** These are left out on purpose:

- Horizontal scaling. The app runs as one process.
- Paid hosting. The whole deployment runs on free plans.
- Many documents per chat. One chat holds one file.
- Email verification, or a document library screen.

---

## 2. What the user can do

| Feature | What the user sees |
|---|---|
| Account | Register and sign in with email and password. The session survives a page reload. |
| Chat | Ask a question. The reply streams in word by word. Chips show which tools the agent used. |
| Upload a file | Attach one PDF, DOCX or text file to a chat. A second upload replaces the first. |
| Remove a file | Close the file chip. The file is deleted for good. |
| Ask about the file | The agent searches the file first and answers from it. |
| Ask about a link | Paste a URL. The agent reads the page. |
| Ask about current events | The agent searches the web. |
| Voice input | Speak. The words appear live in the input box. |
| Chat history | Chats are saved. Open one later and continue. |
| Long chats | Old messages get summarized. A chip says "summarizing" while it happens. |
| Model picker | Choose from a short list of tested models. |
| Theme | Light or dark. |
| Cold start | The backend sleeps when idle. A waiting screen shows until it wakes. |

Limits the user can hit:

- Two chats per account. A third is refused with a message. The server never
  deletes an old chat to make room.
- Ten replies per minute and five uploads per ten minutes, per account.
- Two hundred model calls per day, shared by the whole deployment.

---

## 3. System shape

```
 Browser (React)
   │  HTTP + SSE, through the same origin
   │  WebSocket, direct to the backend (voice only)
   ▼
 Backend (FastAPI, one process)
   api ──┐
         ├──▶ application ──▶ domain
 adapters┘
   │
   ├── Postgres  users, chats, messages, documents, agent checkpoints
   ├── Qdrant    file chunks as vectors
   ├── Together  the chat model
   ├── Gemini    embeddings
   ├── Tavily    web search
   └── Deepgram  speech to text
```

**SSE** means server-sent events. The server pushes a stream of small messages
over one HTTP response. The chat reply uses it.

**Where it runs:**

| Piece | Host |
|---|---|
| Frontend | Vercel |
| Backend | Render, free web service, built from `backend/Dockerfile` |
| Postgres | Neon |
| Qdrant | Qdrant Cloud |

---

## 4. Core design choices

Each choice names the problem it solves.

### Backend structure

- **Onion architecture.** Code sits in four layers: domain, application,
  adapters and api. Imports only point inward. `lint-imports` enforces this in
  the build.
- **Domain is plain Python.** It has no framework imports. It holds entities,
  errors, guardrail detectors and the routing regex.
- **Use cases depend on ports.** A **port** is an interface the application
  layer defines. Ports are `typing.Protocol` classes. Adapters match them by
  shape, never by inheritance. `mypy --strict` proves each fake still matches.
- **One composition root.** `app/composition.py` is the only file that builds
  clients and wires everything. `app/settings.py` is the only file that reads
  the environment. Nothing is built at import time.
- **One transaction per use case call.** Each call opens its own unit of work.
  Sessions are never shared between requests.
- **One place for error mapping.** Use cases raise domain errors.
  `app/api/errors.py` turns them into status codes. A 5xx never shows
  internals.
- **One place for the wire format.** `GenerateReply` yields event objects. Only
  `sse.py` knows the SSE format. Voice uses a transport port, so the use case
  never names WebSockets.

### The agent

- **LangGraph agent with three tools.** `retrieve_documents`,
  `fetch_web_pages` and `search_web`. Postgres stores its state between turns.
- **Tool rules live in code.** `ToolRegistry.invoke` is the single choke point
  for every tool call. It enforces two rules:
  1. On a question about the user's own files, web search waits until
     retrieval has run. An empty retrieval opens the search. So the model
     still decides whether to go to the web. Only the order is fixed.
  2. On an account with no files, retrieval is refused.

  Why: the default model picks the right tool only about a third of the time.
  A rule that lives only in a prompt is a rule the next model ignores.
- **The agent knows what the user uploaded.** Each upload gets a short summary,
  called the **digest**. Every turn adds the digest to the user message. An
  account with no files is told so in one line. The digest informs the choice.
  The routing rules make it binding.
- **Request facts come from the server, never the model.** The owner, the chat
  id and "is this about their files" ride the run config. The model cannot set
  them.
- **Long chats are summarized.** Old history is condensed by a cheaper model.
  Big tool outputs are condensed too. Token budgets bound the cost of every
  loop.
- **The last loop gets no tools.** When the agent hits its loop limit, it must
  answer in text. So the limit never leaves the user with an empty reply.
- **Provider is a setting.** `llm_provider` is a string. Moving from Groq to
  Together changed two environment variables and no code.

### Documents

- **A document belongs to one chat, not to the account.** Every search filters
  on owner and chat. The owner filter is the security boundary. The chat filter
  stops one chat from answering out of another chat's file.
- **"Deleted" means three things together.** The vectors, the stored file and
  the database row. One use case, `DeleteDocument`, does all three. Every other
  path calls it.
- **A replacement upload is safe.** The old file is removed only after the new
  one is stored. A failed upload never costs the user the file they had.

### Security

- **Untrusted text is fenced.** Tool results and digests are wrapped in
  `<untrusted_content>` tags. This defends against **indirect prompt
  injection**, which is instructions hidden in a web page or a file.
- **Guardrails run before the model.** They redact secrets and flag
  injection in user input. The scans have a size cap, because a long regex scan
  would stall every open stream on the one worker.
- **Two tokens.** The access token lasts 15 minutes and needs no database
  lookup. The refresh token lasts 14 days, lives in an httpOnly cookie, rotates
  on every use and can be revoked. Reusing a revoked token kills the whole
  session family.
- **The access token lives in memory only.** Never in `localStorage`.
  Otherwise a script could steal it.
- **The WebSocket checks its own origin.** CORS does not run on a WebSocket
  handshake. So the route checks `Origin` itself and refuses a missing one.

### Frontend

- **Feature slices.** `chat`, `auth` and `theme` each own their components,
  hooks, API calls and types. Other code imports a feature through its
  `index.ts`.
- **One token file.** `globals.css` is the only file allowed a literal color.
- **One motion file.** Every animation timing lives in `lib/motion.ts`.
- **One refresh at a time.** `authorizedFetch` refreshes the token behind a
  single shared promise. Two refreshes at once would look like token theft and
  sign the user out everywhere.
- **Same origin for HTTP.** `frontend/middleware.ts` forwards API calls to the
  backend. The browser sees one origin, so the refresh cookie stays
  first-party. The WebSocket cannot pass through that hop, so it connects
  directly.

### Deployment and schema

- **Migrations own the schema everywhere.** `alembic upgrade head` is the only
  thing that creates tables. The app never calls `create_all`. LangGraph's
  checkpoint tables are the one exception. The app creates those at startup.
- **Free hosting over a paid box.** Render's free plan cost three things: the
  edge rate limit, a persistent disk and a scheduler. Each loss is written
  down under Known limits.
- **Liveness and readiness are different.** `/health` never touches a
  dependency. `/ready` checks Postgres and Qdrant and answers 503 when one is
  down. Tracing is never part of `/ready`.
- **Observability costs nothing when off.** Every request gets an id that shows
  in logs, error bodies and Sentry. Sentry and Langfuse run only when their
  keys are set.

---

## 5. Rules that must always hold

These are the project's invariants. Breaking one is a bug, even if tests pass.

1. Imports only point inward. `composition.py` is the only exception.
2. Only `settings.py` reads the environment.
3. Every tool call goes through `ToolRegistry.invoke`.
4. Every vector search filters on owner and chat. A missing chat id returns
   nothing. It is never a wildcard.
5. Deleting a document removes vectors, file and row together.
6. The server never deletes a user's work to make room.
7. Every authenticated frontend call goes through `authorizedFetch`.
8. Only Alembic creates tables.
9. The app refuses to start against a schema older than its code.
10. A database schema error is never turned into a 200.
11. `DATABASE_URL` and `DATABASE_MIGRATION_URL` name the same database.
12. A provider's error text never reaches the user. It goes to the log and the
    trace only.
13. A config file in the repo is not proof of what the platform runs. Check
    the live service.
14. When a file's job moves, every comment that names the file moves too.

---

## 6. Rules the code does not meet yet

The rules above are the target. Four of them are not in the code yet:

| Rule | Where the code falls short |
|---|---|
| 9. Refuse an old schema | Nothing compares `alembic_version` to the code's head at startup. |
| 10. No schema error becomes a 200 | `_owned_documents` in `generate_reply.py` catches every exception and returns `None`. |
| 11. Same database for both URLs | Nothing checks the pair at startup. |
| 12. No provider text to the user | `_friendly_error` rewrites two cases. Every other error passes through as the provider wrote it. |

---

## 7. Budgets and numbers

| Setting | Value | Why |
|---|---|---|
| `max_conversations_per_user` | 2 | Each chat keeps a file embedded. This caps storage per account. |
| `chat_rate_limit_requests` | 10 per 60 s | Lets a reviewer test quickly. Stops a script. |
| `upload_rate_limit_requests` | 5 per 600 s | Same reason. |
| `global_daily_call_budget` | 200 per day | Registration has no CAPTCHA. This caps total spend across all accounts. |
| `auth_rate_limit_attempts` | 5 per 300 s | Slows password guessing. |
| `agent_max_tool_iterations` | 8 | Stops a looping agent. |
| `agent_history_token_budget` | 12,000 | Above this, old history gets summarized. |
| `agent_recent_token_budget` | 6,000 | Recent messages kept word for word. |
| `chat_digest_max_documents` | 6 | Most digests added to one turn. |
| `chunk_size` | 1,400 | Characters per indexed chunk. |
| Access token | 15 min | Short, so a stolen one expires fast. |
| Refresh token | 14 days | Rotated on every use. |

---

## 8. Known limits

These are accepted for now. Each has a reason.

- **One process only.** The rate limiter keeps its counts in memory. More
  instances need a shared store such as Redis first.
- **Uploads sit on a disk that gets wiped.** This works only because nothing
  reads the file after indexing. Any feature that needs the file back needs
  object storage first.
- **No edge rate limit.** The daily budget is the only ceiling a new account
  cannot get around. Cloudflare in front of a custom domain would add one back.
- **Cold starts.** The free backend sleeps after about 15 minutes. The next
  visitor waits 30 to 60 seconds.
- **The token cleanup job does not run.** It needs a scheduler, and Render
  cron needs a paid plan. The `refresh_tokens` table slowly grows.
- **Weak default model.** The default model picked the right tool 36% of the
  time in evals. The routing rules make that survivable.
- **`agent_max_request_tokens` is a guess.** It has not been confirmed against
  a real provider error.
- **Timestamps have no time zone.** `created_at` and `updated_at` drop the
  offset. Fix this before there is real data.
- **The proxy route list has no check.** A route missing from the `matcher` in
  `middleware.ts` returns `index.html` with a 200 instead of an error.
- **A failed replacement can leave an old file behind.** The log says
  `Could not replace document`. Nothing in the UI shows it.
- **The summarizer can fail quietly.** Then old messages are dropped, and the
  chat forgets its past. The log says `Condenser failed on`.
- **No email verification, no metrics, no component tests.**

---

## 9. How quality is checked

| Check | What it proves | When it runs |
|---|---|---|
| `lint-imports` | Layers point inward | CI, on backend changes |
| `mypy --strict` | Fakes still match their ports | CI, on backend changes |
| `pytest` | Domain, use cases and API work, with no infrastructure | CI, on backend changes |
| `pytest -m integration` | Real Postgres and Qdrant behaviour, and the migration history | CI, separate job |
| Red-team tests | Guardrails catch known attacks. Zero API calls. | CI, on backend changes, inside `pytest` |
| Evals, `ci` tag | The real model picks the right tools and cites the right source | CI, on pull requests that touch a prompt, tool, guardrail or dataset |
| Vitest | Frontend logic | CI, on frontend changes |

The eval judge's scores are reported but never block a merge. A 0 to 1 score
from a model is too noisy to gate on.

---

## 10. Where the details live

This file does not replace these. The project depends on them.

| File | What it holds |
|---|---|
| `README.md` | Front page for visitors. How to run it. |

| `backend/README.md` | Every endpoint and its use case. Backend limits and decisions. |
| `frontend/README.md` | Feature slices, the auth flow, the cold-start gate. |
| `CLAUDE.md` | Working rules for writing code in this repo, with their reasons. |
| `backend/.env.example` | Every setting, with a comment on each. |
| `render.yaml` | The backend deployment on Render. |
| `.claude/skills/` | How to write evals and motion code here. |
