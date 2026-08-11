# Frontend

React 19 + Vite + TypeScript client for Talk to the Web. See the root
`README.md` for the whole-picture view (product, deployment); this file is
frontend internals.

## Commands

```bash
npm install
npm run dev        # :5173, proxies /auth /generate /upload /conversations /models /ws to :8000
npm run build      # tsc -b && vite build
npm run lint        # oxlint
npm run typecheck   # tsc -b, same check the build runs
npm test            # vitest run
npm run test:watch  # vitest
npm test -- src/lib/session.test.ts    # one file
```

Tests are Vitest, colocated as `src/**/*.test.ts`, and run in the `node`
environment — no jsdom is installed, so a component test needs both the
package and `environment: 'jsdom'` in the `test` block of `vite.config.ts`.
`lib/session.ts` keeps the access token and the in-flight refresh in module
variables, so its tests call `vi.resetModules()` and re-import the module
for each case rather than resetting that state from outside.

## Architecture

```
src/
  app/        shell (App.tsx)
  features/
    chat/     components/ hooks/ api.ts types.ts index.ts (barrel)
    auth/     same shape
    theme/    same shape
  components/ui/   shared primitives (Button, IconButton, Mark, ErrorBoundary)
  lib/        http.ts  session.ts  motion.ts  conversation.ts
  styles/     globals.css (token contract)
```

- Feature-sliced: each feature owns its `components/`, `hooks/`, `api.ts`,
  `types.ts`, and an `index.ts` barrel. Other code imports a feature through
  its barrel, not by reaching into its internals.
- `src/styles/globals.css` is the token contract and the **only** file under
  `src/` permitted to contain a literal hex value — everything else
  references its custom properties. Theming is `data-theme` on the root; see
  the `frontend-motion` skill before touching animation.
- All Motion springs and timings live in `src/lib/motion.ts` — never
  hand-roll a transition config inline in a component.
- API responses are narrowed at the boundary with the helpers in
  `lib/http.ts` (`ApiError`, `requireStringFields`), then trusted downstream.

## Features

| Feature | Where | Notes |
|---|---|---|
| Chat (streamed replies) | `features/chat` (`useChat`, `api.ts`) | Consumes the backend's SSE stream from `POST /generate/text/` |
| Conversation list / switch / delete | `features/chat` (`useConversations`, `Sidebar`) | Delete is a `POST`, not a `DELETE` — see below |
| Document upload / list / delete | `features/chat` (`useDocuments`, `useFileUpload`, `DocumentManager`) | Drives `/upload/*` and `/documents/*` |
| Model picker | `features/chat` (`useModels`) | Backed by `GET /models/` |
| Live voice input | `features/chat` (`useVoiceInput`) | Connects a WebSocket directly to the backend (see below), streams mic audio, renders partial/final transcripts |
| Auth (register/login/session) | `features/auth` (`AuthProvider`, `AuthGate`, `AuthForm`, `useAuth`) | Access token in memory only; see Key decisions |
| Theme toggle | `features/theme` | Light/dark via `data-theme` |
| Markdown rendering | `features/chat/markdown.ts` | `react-markdown` + `remark-gfm`, has its own test file |

## Operational concerns

- **Auth token refresh is single-flighted.** Every authenticated call goes
  through `authorizedFetch` in `lib/session.ts`, which attaches the bearer
  token and, on a 401, refreshes **once** behind a single in-flight promise
  before retrying. This isn't an optimization: rotation spends the presented
  refresh token, so two concurrent refreshes look like token reuse and
  revoke every session the user has. Any new authenticated call must go
  through this helper, not a bare `fetch`.
- **Route parity between `nginx.conf` and `vercel.json`.** Both list the
  same API path prefixes for the reverse proxy / rewrites. Adding a backend
  route to one and not the other is how a call quietly starts returning
  `index.html` with a 200 instead of an error.
- **The WebSocket is the one call that can't be proxied.** Vercel does not
  proxy an `Upgrade` handshake, so `useVoiceInput` reads `VITE_WS_URL` and
  connects to the backend directly, falling back to same-origin when unset
  (what `npm run dev` and the nginx parity harness use). This makes it the
  one route the browser's CORS/same-origin model doesn't cover — the backend
  enforces `ALLOWED_WEBSOCKET_ORIGINS` itself instead.

## Known limits

- A reload always starts with no access token — `AuthProvider` calls
  `/auth/refresh` on mount to get one back, so there's a brief unauthenticated
  window on every full page load.
- No component tests yet — the Vitest setup runs in the `node` environment
  only; adding one means adding jsdom and the `environment: 'jsdom'` block
  first.
- The voice input path depends on a direct, cross-origin WebSocket in
  production; if `VITE_WS_URL` and the backend's `ALLOWED_WEBSOCKET_ORIGINS`
  drift out of sync, voice input fails silently at the handshake rather than
  with a visible error in the chat UI.

## Key decisions

- **Access token in a module variable, never `localStorage`.** The refresh
  token is an httpOnly cookie precisely so a script can't reach it; storing
  the access token in `localStorage` would undo that protection for the
  half that's left.
- **`/auth` proxied same-origin everywhere** — dev (Vite proxy), and
  production (`vercel.json` rewrites, `nginx.conf`) — so the refresh cookie
  stays first-party. A cross-site cookie would need `SameSite=None`, which a
  plain-http dev origin can't accept.
- **Conversation deletion is a `POST`, not a `DELETE`.** CORS only allows
  GET, POST and OPTIONS here. It was originally POST to support
  `sendBeacon` on unload; that use case went away with accounts (a beacon
  can't carry an `Authorization` header, and an owned conversation should
  outlive the tab), but the method stayed POST since the rewrite/CORS setup
  already depends on it.
- **No inline transition configs.** Centralizing springs/timings in
  `lib/motion.ts` is what keeps motion consistent across features instead of
  each component inventing its own feel.
