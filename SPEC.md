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

**Why it exists.** It is a portfolio project. A reviewer opens the live demo,
tries it, then reads the code.

**Goals:**

1. **Layers a machine checks.** A wrong import fails the build.
2. **Tool rules in code.** The prompt suggests, the code enforces. Untrusted
   text is fenced. Evals measure the agent's choices.
3. **Written limits.** Every limit is listed in section 8. Every paid call has
   a ceiling. A failure the UI does not show is listed there too.

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

Each row names a choice, the option it replaced or ruled out, and why.
`CLAUDE.md` holds the full reasoning and the file names behind each row.

| Choice | Instead of | Why |
|---|---|---|
| Onion layers, checked by `lint-imports` | Layering kept by review | A wrong import fails the build. |
| Ports are `typing.Protocol`, matched by shape | Base classes that adapters inherit | Adapters never import the application. `mypy --strict` still proves each fake matches. |
| A unit of work per use-case call | One session shared across calls | A session is not safe to share between concurrent requests. |
| Domain errors mapped in `app/api/errors.py` | `HTTPException` raised in use cases | Use cases stay free of HTTP. A 5xx never shows internals. |
| Wire format only in `sse.py` | Use cases that yield SSE text | The use case yields events. Voice uses a transport port the same way. |
| Tool rules enforced in `ToolRegistry.invoke` | Tool rules in the prompt only | The default model picks the right tool about a third of the time. |
| Upload digest added to each user turn | Digest in the system prompt | The system prompt is written once per chat. Files arrive between turns. |
| "No files" said in one line | Saying nothing | Silence made the model try a retrieval on every turn of an empty account. |
| "About their files" decided once per request | Re-read from history by the tool node | Summarizing can replace the history. A gate reading a lossy record fails open. |
| Owner and chat id in the run config | Values the model passes | The model must never name whose data it reads. |
| Summarizing shown as a chip | Condensing in silence | A long condense looks like a stalled reply. |
| The last loop gets no tools | Ending the run at the loop limit | The limit must never leave the user with an empty reply. |
| A document belongs to one chat | Documents shared across the account | A question in one chat must not be answered from another chat's file. |
| Deletion goes through `DeleteDocument` | A database cascade | A cascade cannot reach Qdrant or the disk. |
| Old file removed after the new one is stored | Removing it first | A failed upload must not cost the user the file they had. |
| Indexing and deletion share a `DocumentLock` | Fire-and-forget indexing | A delete that won the race left vectors no row could ever remove. |
| A third chat is refused | Evicting the oldest chat | Deleting someone's work is not the server's call. |
| Untrusted text fenced in `<untrusted_content>` | Passing tool output as is | It defends against **indirect prompt injection**, instructions hidden in a page or file. |
| Guardrail scans have a size cap | Scanning all the text | A regex cannot time out, and a long scan stalls every stream on the one worker. |
| Access token in memory only | `localStorage` | A script could read it. The refresh cookie is httpOnly for the same reason. |
| One refresh at a time in `authorizedFetch` | Each call refreshing on its own | Two refreshes look like token reuse and sign the user out everywhere. |
| The WebSocket checks `Origin` itself | Trusting `CORSMiddleware` | CORS does not run on a WebSocket handshake. |
| API proxy in `frontend/middleware.ts` | Rewrites in `vercel.json` | `vercel.json` has no environment variables, so the backend host would be checked in. |
| Alembic creates every table | `create_all` at startup | One path to the schema in every environment. LangGraph's checkpoint tables are the one exception. |
| Render free plan, one service | EC2 with nginx | No hosting cost. Section 8 lists what the free plan took away. |
| `/ready` checks Postgres and Qdrant only | Tracing on `/ready` too | A tracing outage must not return 503. |
| The LLM provider is a setting | Provider-specific code | Moving from Groq to Together changed two variables and no code. |

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

Two working habits in `CLAUDE.md` sit next to these. They are about how we
work, not what the system does: check the live service before trusting a
config file, and move every comment that names a file when its job moves.

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
