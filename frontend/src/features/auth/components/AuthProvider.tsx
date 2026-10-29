import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import { getAccessToken, refreshSession, subscribeToSession } from '../../../lib/session'
import { signOut as requestSignOut, submitCredentials } from '../api'
import { AuthContext, type AuthValue } from '../hooks/useAuth'
import type { AuthMode, Credentials, SessionUser } from '../types'

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<SessionUser | null>(null)
  const [restoring, setRestoring] = useState(true)

  /**
   * Restore the session from the refresh cookie on first paint.
   *
   * A failure just means "not signed in".
   */
  useEffect(() => {
    let cancelled = false

    refreshSession()
      .then((session) => {
        if (!cancelled) setUser(session.user)
      })
      .catch(() => {
        if (!cancelled) setUser(null)
      })
      .finally(() => {
        if (!cancelled) setRestoring(false)
      })

    return () => {
      cancelled = true
    }
  }, [])

  /**
   * Sign out when `authorizedFetch` clears the token after a failed refresh.
   */
  useEffect(
    () =>
      subscribeToSession(() => {
        if (getAccessToken() === null) setUser(null)
      }),
    [],
  )

  const submit = useCallback(async (mode: AuthMode, credentials: Credentials) => {
    const session = await submitCredentials(mode, credentials)
    setUser(session.user)
  }, [])

  const signOut = useCallback(async () => {
    await requestSignOut()
    setUser(null)
  }, [])

  const value = useMemo<AuthValue>(
    () => ({
      status: restoring ? 'restoring' : user ? 'signed-in' : 'anonymous',
      user,
      submit,
      signOut,
    }),
    [restoring, signOut, submit, user],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
