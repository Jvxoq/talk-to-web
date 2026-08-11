/** Thrown for any non-2xx response, so callers can branch on status. */
export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
    /**
     * How long to wait before this can succeed, on a 429. Optional because it
     * is the only failure that has an answer: everything else here fails again
     * at the same speed no matter how long you wait.
     */
    public readonly retryAfterSeconds?: number,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

/** True when a rejection is just an aborted request, not a real failure. */
export function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}

export function messageFrom(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback
}

/**
 * Narrows an unknown JSON body to a record with the given string fields.
 * Responses are untrusted input — parse once at the boundary, then trust the
 * type downstream rather than optional-chaining through the app.
 */
export function requireStringFields<K extends string>(
  raw: unknown,
  keys: readonly K[],
  context: string,
): Record<K, string> {
  if (typeof raw !== 'object' || raw === null) {
    throw new ApiError(0, `${context}: expected an object`)
  }
  const record = raw as Record<string, unknown>
  const out = {} as Record<K, string>
  for (const key of keys) {
    const value = record[key]
    if (typeof value !== 'string') {
      throw new ApiError(0, `${context}: missing "${key}"`)
    }
    out[key] = value
  }
  return out
}
