import { createContext, useContext } from 'react'
import type { AuthMode, Credentials, SessionUser } from '../types'

/** `restoring` is the state a reload starts in, before the cookie is tried. */
export type AuthStatus = 'restoring' | 'anonymous' | 'signed-in'

export interface AuthValue {
  status: AuthStatus
  user: SessionUser | null
  submit: (mode: AuthMode, credentials: Credentials) => Promise<void>
  signOut: () => Promise<void>
}

/**
 * The context and its hook, kept apart from the provider component.
 *
 * A module that exports both a component and a hook defeats React Fast Refresh
 * — it cannot tell which half changed, so it reloads the whole tree and drops
 * state on every edit. The provider lives in `components/AuthProvider.tsx`.
 */
export const AuthContext = createContext<AuthValue | null>(null)

export function useAuth(): AuthValue {
  const value = useContext(AuthContext)
  if (value === null) {
    throw new Error('useAuth must be used inside an <AuthProvider>')
  }
  return value
}
