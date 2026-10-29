import { ApiError } from './http'

const AUTH_URL = import.meta.env.VITE_AUTH_URL ?? '/auth'

/**
 * The access token, kept only in this module variable.
 *
 * Not `localStorage`, so a script cannot read it. A reload starts empty, and
 * `restore()` gets a new one from the refresh cookie.
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
 * `credentials: 'include'` sends the cookie. Plain `fetch`, since
 * `authorizedFetch` would loop.
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
 * The refresh in flight, if any.
 *
 * Shared, because two concurrent refreshes look like token reuse and revoke
 * every session.
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
 * `fetch` with the bearer token, plus one retry after a refresh on a 401.
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
    // The refresh failed, so clear the session here for every caller.
    clearAccessToken()
    return response
  }
}
