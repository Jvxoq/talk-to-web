import { clearConversationId } from '../../lib/conversation'
import { ApiError } from '../../lib/http'
import {
  authEndpoint,
  clearAccessToken,
  parseSession,
  setAccessToken,
  type Session,
} from '../../lib/session'
import type { AuthMode, Credentials } from './types'

/**
 * Posts credentials and returns the session.
 *
 * `credentials: 'include'` lets the browser store the refresh cookie.
 */
export async function submitCredentials(
  mode: AuthMode,
  { email, password }: Credentials,
): Promise<Session> {
  const response = await fetch(authEndpoint(mode === 'login' ? 'login' : 'register'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ email, password }),
  })

  if (!response.ok) {
    throw new ApiError(response.status, await detailFrom(response, mode))
  }

  const session = parseSession(await response.json())
  setAccessToken(session.accessToken)
  return session
}

/**
 * Ends the session on the server, then locally. The local part always runs.
 */
export async function signOut(): Promise<void> {
  try {
    await fetch(authEndpoint('logout'), { method: 'POST', credentials: 'include' })
  } finally {
    clearAccessToken()
    // The pinned thread belongs to the account that left, so drop it.
    clearConversationId()
  }
}

const FALLBACKS: Record<number, string> = {
  401: 'Email or password is incorrect.',
  409: 'That email is already registered.',
  429: 'Too many attempts. Please wait a moment.',
}

/**
 * The message to show: the server's own wording, except on a 5xx.
 */
async function detailFrom(response: Response, mode: AuthMode): Promise<string> {
  if (response.status >= 500) {
    return mode === 'login' ? 'Could not sign in. Please try again.' : 'Could not sign up.'
  }

  const body: unknown = await response.json().catch(() => null)
  const detail =
    typeof body === 'object' && body !== null && 'detail' in body
      ? (body as { detail: unknown }).detail
      : null

  if (typeof detail === 'string' && detail) return detail
  return FALLBACKS[response.status] ?? `Request failed with status ${response.status}`
}
