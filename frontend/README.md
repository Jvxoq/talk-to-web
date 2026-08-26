# Frontend

React 19 + Vite + TypeScript client for Talk to the Web. See the root
`ARCHITECTURE.md` for the whole-picture view (product, deployment); this file is
frontend internals.

## Commands

```bash
npm install
npm run dev        # :5173, proxies /auth /generate /upload /documents /conversations /models /ws to :8000
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
| Summarizing notice | `features/chat` (`summarizing.ts`, `MessageBubble`) | The `summarizing` frame becomes one chip beside the tool chips, so the silent pause while the thread is condensed has a reason on screen. One slot per turn, overwritten |
| Conversation list / switch / delete | `features/chat` (`useConversations`, `Sidebar`) | Delete is a `POST`, not a `DELETE` — see below. A 409 from "new chat" is the per-account cap, shown as `sidebar-error` under the button. The bootstrap also preloads the pinned conversation's transcript, see below |
| Document upload | `features/chat` (`useFileUpload`, `Composer`) | `POST /upload/file/`, naming the conversation. The chip's close button calls `POST /documents/{id}/delete`, which removes the file for good. Switching threads drops the chip without deleting anything |
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
- **The conversation bootstrap runs two requests in parallel.** The pinned
  conversation id is read synchronously from `localStorage`, so
  `useConversations` fires `listConversations()` and `getConversation(pinned)`
  together and hands the transcript to `useChat` as `preloaded`. Doing them in
  sequence was the whole of the "it takes a while before the chat shows up"
  delay on load. A stale pin makes the second request wasted work, which is
  much cheaper than two serial round trips on every visit; it resolves to null
  rather than failing the pair. The preload is consumed once per conversation
  id, so switching away and back re-fetches instead of showing a stale
  snapshot.
- **`vercel.json` lists every API path prefix, and the list has to be
  complete.** A backend route missing from the rewrites does not error. It
  falls through to the SPA catch-all and returns `index.html` with a 200.
  It is also the only such list now: `nginx.conf` and the frontend
  `Dockerfile` went with the EC2 deployment, so nothing cross-checks this one.
  Adding a route to the API means adding it here by hand.
- **The WebSocket is the one call that can't be proxied.** Vercel does not
  proxy an `Upgrade` handshake, so `useVoiceInput` reads `VITE_WS_URL` and
  connects to the backend directly, falling back to same-origin when unset
  (what `npm run dev` uses). This makes it the
  one route the browser's CORS/same-origin model doesn't cover — the backend
  enforces `ALLOWED_WEBSOCKET_ORIGINS` itself instead.

## Known limits

- A reload always starts with no access token — `AuthProvider` calls
  `/auth/refresh` on mount to get one back, so there's a brief unauthenticated
  window on every full page load.
- No component tests yet — the Vitest setup runs in the `node` environment
  only; adding one means adding jsdom and the `environment: 'jsdom'` block
  first.
- **No document list UI.** The attachment chip can remove the file it names
  (`POST /documents/{id}/delete`), and a thread holds one file at a time, so
  the chip is the whole document surface. `GET /documents/` is still served
  and nothing in `src/` calls it.
- **Removing an attachment is optimistic and one-way.** The chip disappears
  before the request answers and does not come back if it fails. The failure
  is shown in the chip's place instead, because re-showing a chip invites the
  user to keep pressing a button that already did what it could.
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
  production (`vercel.json` rewrites) — so the refresh cookie
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
