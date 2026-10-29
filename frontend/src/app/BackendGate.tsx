import { useCallback, useEffect, useState, type ReactNode } from 'react'
import { motion } from 'motion/react'
import { buttonClass, PulseBlocks } from '../components/ui'
import { easeStandard, springs, timing } from '../lib/motion'
import { waitForBackend } from '../lib/wake'

/**
 * Delay before showing the wait screen, so an awake backend never flashes it.
 */
const REVEAL_DELAY_MS = 900

type GateStatus = 'checking' | 'awake' | 'unreachable'

/**
 * Holds the app back until the backend answers.
 *
 * Wraps `AuthProvider`, whose first call would fail on a sleeping backend.
 */
export function BackendGate({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<GateStatus>('checking')
  const [revealed, setRevealed] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  // Bumped by Retry to re-run the effect.
  const [run, setRun] = useState(0)

  const retry = useCallback(() => setRun((n) => n + 1), [])

  useEffect(() => {
    const controller = new AbortController()
    setStatus('checking')
    setRevealed(false)
    setElapsed(0)

    const reveal = setTimeout(() => setRevealed(true), REVEAL_DELAY_MS)
    const tick = setInterval(() => setElapsed((seconds) => seconds + 1), 1000)

    void waitForBackend({ signal: controller.signal }).then((result) => {
      // Stop here: the gate stays mounted, so cleanup never runs.
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
      transition={{ duration: timing.standard, ease: easeStandard }}
      // `role="alert"` is assertive. An `aria-live` would make it polite.
      role={failed ? 'alert' : undefined}
      aria-live={failed ? undefined : 'polite'}
      aria-busy={!failed}
    >
      {/* Hidden once the wait has failed: nothing is loading any more. */}
      {!failed && <PulseBlocks />}

      <h2 className="backend-gate__title">
        {failed ? 'The backend did not answer' : 'Waking the backend'}
      </h2>

      <p className="backend-gate__body">
        {failed
          ? 'It may still be starting, or it may be down. Try again.'
          : 'This runs on a free plan and goes to sleep. It takes about 30-60 seconds.'}
      </p>

      {failed ? (
        <motion.button
          type="button"
          className={buttonClass('primary')}
          onClick={retry}
          whileHover={{ x: -2, y: -2 }}
          whileTap={{ x: 0, y: 0 }}
          transition={springs.press}
        >
          Try again
        </motion.button>
      ) : (
        // A counter, not a bar: the wait has no progress to report.
        <p className="backend-gate__elapsed">{elapsed}s</p>
      )}
    </motion.div>
  )
}
