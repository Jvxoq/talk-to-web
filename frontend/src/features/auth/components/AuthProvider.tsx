import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import { getAccessToken, refreshSession, subscribeToSession } from '../../../lib/session'
import { signOut as requestSignOut, submitCredentials } from '../api'
import { AuthContext, type AuthValue } from '../hooks/useAuth'
import type { AuthMode, Credentials, SessionUser } from '../types'

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<SessionUser | null>(null)
  const [restoring, setRestoring] = useState(true)

  /**
   * Rehydrate from the refresh cookie on first paint.
   *
   * The access token lives only in memory, so a reload always starts signed out
   * as far as this app is concerned. The cookie is what remembers, and this is
   * the one call that asks it. A failure is the ordinary "not signed in"
   * answer, not an error worth showing.
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
   * Follow the token when something outside React drops it.
   *
   * `authorizedFetch` clears it when a mid-session refresh fails, and it knows
   * nothing about React. Without this the app would keep rendering a signed-in
   * shell whose every request 401s.
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
