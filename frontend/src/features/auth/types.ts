import type { SessionUser } from '../../lib/session'

export type { SessionUser }

/** Which half of the form is showing. */
export type AuthMode = 'login' | 'register'

export interface Credentials {
  email: string
  password: string
}
