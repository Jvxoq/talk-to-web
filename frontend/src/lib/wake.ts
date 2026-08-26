/**
 * Waiting for a sleeping backend to come back.
 *
 * The free Render instance stops after fifteen minutes without traffic, and the
 * next request pays the container start plus `alembic upgrade head`. Nothing
 * else in the app can tell that apart from being signed out, so the wait
 * happens here, before anything that needs the backend is mounted.
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
 * One probe. True only for a real answer from the real backend.
 *
 * The body is checked, not just the status: a `/health` missing from the proxy
 * list falls through to the SPA catch-all and answers 200 with `index.html`.
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
    // A network error, a timeout, or an HTML body that will not parse. During a
    // cold start all three are the expected answer.
    return false
  }
}

/**
 * Polls `/health` until the backend answers or the deadline passes.
 *
 * Never rejects: the caller renders a screen from the result, and a cold start
 * is not an error. An aborted signal ends the loop as `'unreachable'`.
 */
export async function waitForBackend(
  options: { signal?: AbortSignal; onAttempt?: () => void } = {},
): Promise<WakeResult> {
  const { signal, onAttempt } = options
  const deadline = Date.now() + WAKE_DEADLINE_MS

  while (!signal?.aborted) {
    onAttempt?.()

    // Two reasons to stop a probe: the caller unmounted, or the attempt hung.
    // `AbortSignal.any` folds them into the one signal `fetch` accepts.
    const timeout = AbortSignal.timeout(ATTEMPT_TIMEOUT_MS)
    const attemptSignal = signal ? AbortSignal.any([signal, timeout]) : timeout

    if (await probe(attemptSignal)) return 'awake'
    if (Date.now() >= deadline) return 'unreachable'

    await sleep(RETRY_DELAY_MS, signal)
  }

  return 'unreachable'
}
