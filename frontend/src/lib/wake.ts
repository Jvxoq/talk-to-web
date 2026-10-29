/**
 * Waits for the sleeping free Render backend to wake up.
 *
 * Runs before anything that needs the backend, so a cold start does not look
 * like being signed out.
 */

const HEALTH_URL = '/health'

/** Total time to keep trying before giving up and offering a retry. */
export const WAKE_DEADLINE_MS = 120_000

/** How long one probe may hang before it is abandoned and retried. */
export const ATTEMPT_TIMEOUT_MS = 15_000

/** Gap between a failed probe and the next one. */
const RETRY_DELAY_MS = 2_000

export type WakeResult = 'awake' | 'unreachable'

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const timer = setTimeout(finish, ms)
    signal?.addEventListener('abort', finish, { once: true })

    function finish(): void {
      clearTimeout(timer)
      signal?.removeEventListener('abort', finish)
      resolve()
    }
  })
}

/**
 * One probe. Checks the body too, since a missing proxy route returns `index.html` with 200.
 */
async function probe(signal: AbortSignal): Promise<boolean> {
  try {
    const response = await fetch(HEALTH_URL, { cache: 'no-store', signal })
    if (!response.ok) return false

    const body: unknown = await response.json()
    return (
      typeof body === 'object' &&
      body !== null &&
      (body as Record<string, unknown>).status === 'ok'
    )
  } catch {
    // Network error, timeout or HTML: all expected during a cold start.
    return false
  }
}

/**
 * Polls `/health` until the backend answers or time runs out. Never rejects.
 */
export async function waitForBackend(
  options: { signal?: AbortSignal; onAttempt?: () => void } = {},
): Promise<WakeResult> {
  const { signal, onAttempt } = options
  const deadline = Date.now() + WAKE_DEADLINE_MS

  while (!signal?.aborted) {
    onAttempt?.()

    // Stop on unmount or on a hung attempt.
    const timeout = AbortSignal.timeout(ATTEMPT_TIMEOUT_MS)
    const attemptSignal = signal ? AbortSignal.any([signal, timeout]) : timeout

    if (await probe(attemptSignal)) return 'awake'
    if (Date.now() >= deadline) return 'unreachable'

    await sleep(RETRY_DELAY_MS, signal)
  }

  return 'unreachable'
}
