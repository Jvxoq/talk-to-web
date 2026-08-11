import { ApiError } from './http'

const AUTH_URL = import.meta.env.VITE_AUTH_URL ?? '/auth'

/**
 * The access token, held in a module variable and nowhere else.
 *
 * Deliberately not `localStorage`. The refresh token lives in an httpOnly
 * cookie precisely so that a script running on this page cannot read it; putting
 * the access token in storage would hand a script the other half and undo the
 * split. In memory, an XSS gets at most whatever is live in this tab right now.
 *
 * The cost is that a reload starts with nothing, which is what `restore()`
 * below is for: the cookie survives, so one call to /auth/refresh gets a new
 * access token without the user typing anything.
 */
let accessToken: string | null = null

/** Notified when the token appears or disappears, so React can re-render. */
const listeners = new Set<() => void>()

function publish(token: string | null): void {
  accessToken = token
  listeners.forEach((listener) => listener())
}

export function subscribeToSession(listener: () => void): () => void {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

export function getAccessToken(): string | null {
  return accessToken
}

export function setAccessToken(token: string): void {
  publish(token)
}

export function clearAccessToken(): void {
  publish(null)
}

export interface SessionUser {
  id: number
  email: string
}

export interface Session {
  accessToken: string
  user: SessionUser
}

/** Narrows an untrusted `/auth/*` response to the fields we read. */
export function parseSession(raw: unknown): Session {
  if (typeof raw !== 'object' || raw === null) {
    throw new ApiError(0, 'Session: expected an object')
  }
  const record = raw as Record<string, unknown>
  const user = record.user as Record<string, unknown> | undefined

  if (typeof record.access_token !== 'string') {
    throw new ApiError(0, 'Session: missing "access_token"')
  }
  if (typeof user?.id !== 'number' || typeof user.email !== 'string') {
    throw new ApiError(0, 'Session: missing "user"')
  }

  return { accessToken: record.access_token, user: { id: user.id, email: user.email } }
}

export function authEndpoint(path: string): string {
  const base = AUTH_URL.endsWith('/') ? AUTH_URL.slice(0, -1) : AUTH_URL
  return `${base}/${path}`
}

/**
 * Exchanges the refresh cookie for a new access token.
 *
 * `credentials: 'include'` is what sends the cookie at all — `fetch` omits
 * cookies cross-origin by default, and this is the one call that depends on it.
 * A plain `fetch`, not `authorizedFetch`: this *is* how a token is obtained, so
 * going through the wrapper would be a loop.
 */
async function requestRefresh(): Promise<Session> {
  const response = await fetch(authEndpoint('refresh'), {
    method: 'POST',
    credentials: 'include',
  })

  if (!response.ok) {
    throw new ApiError(response.status, 'Session expired')
  }
  return parseSession(await response.json())
}

/**
 * The refresh in flight, if there is one.
 *
 * Single-flight, and it has to be. Rotation invalidates the token it was given,
 * so two concurrent refreshes would race: the second presents a token the first
 * already spent, the server reads that as reuse, and it revokes every session
 * the user has. Sharing one promise turns a burst of 401s into one rotation.
 */
let inFlight: Promise<Session> | null = null

export function refreshSession(): Promise<Session> {
  inFlight ??= requestRefresh()
    .then((session) => {
      setAccessToken(session.accessToken)
      return session
    })
    .catch((error: unknown) => {
      clearAccessToken()
      throw error
    })
    .finally(() => {
      inFlight = null
    })

  return inFlight
}

/**
 * `fetch`, with the bearer token attached and one silent retry after a refresh.
 *
 * Access tokens are short-lived by design, so a 401 mid-session is the ordinary
 * case rather than an error: refresh once, replay the request, and only give up
 * — signed out — if that fails too. Retrying more than once would loop against a
 * server that simply does not accept us.
 */
export async function authorizedFetch(url: string, init: RequestInit = {}): Promise<Response> {
  const send = (token: string | null): Promise<Response> =>
    fetch(url, {
      ...init,
      headers: {
        ...init.headers,
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
    })

  const response = await send(accessToken)
  if (response.status !== 401) return response

  try {
    const session = await refreshSession()
    return await send(session.accessToken)
  } catch {
    // The refresh failed, so there is no session left to speak of. Cleared here
    // rather than left to the caller: every caller would have to remember, and
    // one that forgot would leave the app rendering a signed-in shell.
    clearAccessToken()
    return response
  }
}
