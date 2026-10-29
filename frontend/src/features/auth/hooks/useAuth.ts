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
 * The context and hook, apart from the provider, so Fast Refresh keeps state.
 */
export const AuthContext = createContext<AuthValue | null>(null)

export function useAuth(): AuthValue {
  const value = useContext(AuthContext)
  if (value === null) {
    throw new Error('useAuth must be used inside an <AuthProvider>')
  }
  return value
}
