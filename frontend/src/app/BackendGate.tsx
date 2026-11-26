import { useCallback, useEffect, useState, type ReactNode } from 'react'
import { motion } from 'motion/react'
import { CircleAlert } from 'lucide-react'
import { buttonClass, Mark, PulseBlocks, WordReveal } from '../components/ui'
import { easings, timing } from '../lib/motion'
import { waitForBackend } from '../lib/wake'

/**
 * How long the screen stays hidden while the first probe runs. An awake backend
 * answers well inside this, and an indicator that appears and vanishes reads as
 * a glitch.
 */
const REVEAL_DELAY_MS = 900

type GateStatus = 'checking' | 'awake' | 'unreachable'

/**
 * Holds the app back until the backend answers.
 *
 * Wraps `AuthProvider` rather than sitting inside it: that provider's mount
 * effect is the first backend call, and a sleeping instance turns it into a
 * sign-in form nobody can get through.
 */
export function BackendGate({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<GateStatus>('checking')
  const [revealed, setRevealed] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  // Bumped by Retry. Changing it re-runs the effect, which is the whole restart.
  const [run, setRun] = useState(0)

  const retry = useCallback(() => setRun((n) => n + 1), [])

  useEffect(() => {
    const controller = new AbortController()
    // `revealed` is not reset here: on a retry the screen is already showing,
    // and hiding it for the reveal delay would blank the page.
    setStatus('checking')
    setElapsed(0)

    const reveal = setTimeout(() => setRevealed(true), REVEAL_DELAY_MS)
    const tick = setInterval(() => setElapsed((seconds) => seconds + 1), 1000)

    void waitForBackend({ signal: controller.signal }).then((result) => {
      // Not left to the cleanup: once the gate opens this stays mounted for
      // the life of the app, so those timers would never stop.
      clearInterval(tick)
      clearTimeout(reveal)

      // Aborted means this effect was cleaned up, so its state is stale.
      if (controller.signal.aborted) return
      setStatus(result === 'awake' ? 'awake' : 'unreachable')
    })

    return () => {
      controller.abort()
      clearTimeout(reveal)
      clearInterval(tick)
    }
  }, [run])

  if (status === 'awake') return <>{children}</>
  if (status === 'checking' && !revealed) return null

  const failed = status === 'unreachable'

  return (
    <motion.div
      className="backend-gate"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: timing.fast, ease: easings.outQuart }}
      // `role="alert"` is already assertive, and a stated `aria-live` would
      // demote the failure to polite.
      role={failed ? 'alert' : undefined}
      aria-live={failed ? undefined : 'polite'}
      aria-busy={!failed}
    >
      <div className="backend-gate__inner">
        <Mark className="backend-gate__mark" />
        <p className="eyebrow">Talk to web</p>

        {/* The word reveal plays once, on arrival. The failure, and every
            retry after it (`run > 0`), swaps the text in place rather than
            replaying the entrance on a screen that is already showing. */}
        {failed || run > 0 ? (
          <h1 className="backend-gate__title">
            {failed ? 'The backend did not answer' : 'Waking the backend'}
          </h1>
        ) : (
          <WordReveal as="h1" className="backend-gate__title" text="Waking the backend" />
        )}

        <p className="backend-gate__body">
          {failed
            ? 'It may still be starting, or it may be down. Try again in a moment.'
            : 'The server runs on a free plan and sleeps when nobody uses it. Your visit is waking it up.'}
        </p>

        {/* A dotted-leader ledger, not a progress bar: the wait has no
            progress to report, only how long it has run. `aria-live="off"`
            keeps the per-second counter from being read out every tick. */}
        <dl className="backend-gate__ledger" aria-live="off">
          <div className="backend-gate__row">
            <dt>Status</dt>
            <dd className={failed ? 'backend-gate__status is-failed' : 'backend-gate__status'}>
              {failed ? (
                <>
                  <CircleAlert size={14} strokeWidth={1.5} aria-hidden="true" />
                  No answer
                </>
              ) : (
                <>
                  <PulseBlocks size="sm" />
                  Waking
                </>
              )}
            </dd>
          </div>
          <div className="backend-gate__row">
            <dt>Elapsed</dt>
            <dd className="backend-gate__num">{elapsed}s</dd>
          </div>
          <div className="backend-gate__row">
            <dt>Usually takes</dt>
            <dd className="backend-gate__num">30–60s</dd>
          </div>
        </dl>

        {failed && (
          <button type="button" className={buttonClass('primary')} onClick={retry}>
            Try again
          </button>
        )}
      </div>
    </motion.div>
  )
}
