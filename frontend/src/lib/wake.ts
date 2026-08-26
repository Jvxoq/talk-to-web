/**
 * Waiting for a sleeping backend to come back.
 *
 * The API runs on a free Render instance, which is stopped after fifteen
 * minutes with no traffic. The next request pays the container start plus the
 * `alembic upgrade head` that runs from `dockerCommand` on every boot, so the
 * first byte can be thirty to sixty seconds away.
 *
 * Nothing else in the app can tell that apart from being signed out:
 * `AuthProvider` calls `/auth/refresh` on mount and reads any failure as
 * "anonymous", which puts a sign-in form in front of someone whose password
 * would fail too. So the wake happens first, here, before anything that needs
 * the backend is mounted.
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
 * The body is checked, not just the status. `/health` is proxied by Vercel and
 * by the Vite dev server, and a missing route falls through to the SPA
 * catch-all, which answers 200 with `index.html` — exactly the trap already
 * documented for `/models` in `vite.config.ts`. A 200 that is not our JSON is
 * a misconfigured proxy, and treating it as awake would hand the user straight
 * back to the failure this module exists to prevent.
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
    // A sleeping instance fails as a network error, a timeout, or an HTML body
    // that will not parse. During a cold start all three are the expected
    // answer rather than something worth reporting.
    return false
  }
}

/**
 * Polls `/health` until the backend answers or the deadline passes.
 *
 * Never throws and never rejects: the caller renders a screen from the result,
 * and a cold start is not an error. An aborted `signal` ends the loop and
 * resolves `'unreachable'`, which the caller ignores because it only aborts on
 * unmount.
 */
export async function waitForBackend(
  options: { signal?: AbortSignal; onAttempt?: () => void } = {},
): Promise<WakeResult> {
  const { signal, onAttempt } = options
  const deadline = Date.now() + WAKE_DEADLINE_MS

  while (!signal?.aborted) {
    onAttempt?.()

    // Two reasons to stop one probe: the caller unmounted, or this attempt hung
    // past its own budget. `AbortSignal.any` folds them into the single signal
    // `fetch` accepts, so a hung request cannot outlive its retry.
    const timeout = AbortSignal.timeout(ATTEMPT_TIMEOUT_MS)
    const attemptSignal = signal ? AbortSignal.any([signal, timeout]) : timeout

    if (await probe(attemptSignal)) return 'awake'
    if (Date.now() >= deadline) return 'unreachable'

    await sleep(RETRY_DELAY_MS, signal)
  }

  return 'unreachable'
}
