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
 * Posts credentials and takes the returned session.
 *
 * `credentials: 'include'` is what lets the browser *store* the refresh cookie
 * the response sets — without it the Set-Cookie is discarded on a cross-origin
 * response, and the user is signed out the moment they reload.
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
 * Ends the session on the server, then locally.
 *
 * The local half runs whether or not the request worked. A user who clicked
 * sign out must end up signed out of this browser regardless of what the
 * network did — and the refresh cookie is the server's to clear.
 */
export async function signOut(): Promise<void> {
  try {
    await fetch(authEndpoint('logout'), { method: 'POST', credentials: 'include' })
  } finally {
    clearAccessToken()
    // The pinned conversation belongs to the account that just left. Left in
    // place it outlives the session in `localStorage`, so the next person to
    // sign in on this browser starts by asking for a thread that is not
    // theirs. They are refused, but a sign-in should not open with a failed
    // request for a stranger's conversation in the first place.
    clearConversationId()
  }
}

const FALLBACKS: Record<number, string> = {
  401: 'Email or password is incorrect.',
  409: 'That email is already registered.',
  429: 'Too many attempts. Please wait a moment.',
}

/**
 * The message to show, preferring the server's own wording.
 *
 * The backend writes these for people — "Password rejected: it must be at least
 * 12 characters" — so echoing it beats a generic string. A 5xx is the exception:
 * its detail is a fixed placeholder, and the status is all we really know.
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
