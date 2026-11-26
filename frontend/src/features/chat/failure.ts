import type { FailureCode } from './types'

const MESSAGES: Record<FailureCode, string> = {
  timeout: 'The assistant took too long to answer. Please try again.',
  overloaded: 'The assistant is busy right now. Please try again in a moment.',
  unavailable: 'The assistant is unavailable right now. Please try again later.',
  internal: 'Something went wrong. Please try again.',
}

function isFailureCode(value: unknown): value is FailureCode {
  return typeof value === 'string' && Object.hasOwn(MESSAGES, value)
}

/**
 * Narrows one untrusted `error` SSE frame payload to a failure code, at the
 * boundary. The backend sends `{ code }` and never the error's own
 * text. Anything else (an older backend's bare string, a code this build does
 * not know yet) reads as `internal`, so no server text is ever shown.
 */
export function parseFailure(raw: unknown): FailureCode {
  if (typeof raw !== 'object' || raw === null) return 'internal'
  const { code } = raw as Record<string, unknown>
  return isFailureCode(code) ? code : 'internal'
}

/** What the reader sees for a failed reply. The copy lives here, not on the server. */
export function failureMessage(code: FailureCode): string {
  return MESSAGES[code]
}
