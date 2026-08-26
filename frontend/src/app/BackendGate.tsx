import { useCallback, useEffect, useState, type ReactNode } from 'react'
import { motion } from 'motion/react'
import { buttonClass, PulseBlocks } from '../components/ui'
import { easeStandard, springs, timing } from '../lib/motion'
import { waitForBackend } from '../lib/wake'

/**
 * How long the screen stays hidden while the first probe runs.
 *
 * An awake backend answers `/health` in well under this, so an ordinary load
 * never shows the screen at all. Same reasoning as the held-back `delay` on
 * `.auth-restoring` in `AuthGate`: an indicator that appears and vanishes reads
 * as a glitch.
 */
const REVEAL_DELAY_MS = 900

type GateStatus = 'checking' | 'awake' | 'unreachable'

/**
 * Holds the app back until the backend answers.
 *
 * Wrapped around `AuthProvider` rather than inside it, because that provider's
 * mount effect is the first backend call the app makes and a sleeping instance
 * turns it into a sign-in form nobody can get through. Once this gate opens,
 * everything downstream can assume the backend is up, exactly as it did before
 * the free plan started sleeping.
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
    setStatus('checking')
    setRevealed(false)
    setElapsed(0)

    const reveal = setTimeout(() => setRevealed(true), REVEAL_DELAY_MS)
    const tick = setInterval(() => setElapsed((seconds) => seconds + 1), 1000)

    void waitForBackend({ signal: controller.signal }).then((result) => {
      // Stopped here rather than left to the cleanup below: once the gate
      // opens this component stays mounted for the life of the app, so timers
      // only the cleanup clears would keep running, and re-rendering, forever
      // behind a screen nobody is looking at any more.
      clearInterval(tick)
      clearTimeout(reveal)

      // Aborted means this effect was cleaned up — under StrictMode's double
      // mount that is the first run, whose state must not reach the second.
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
      // `role="alert"` already implies an assertive live region, so spelling
      // out `aria-live` as well would quietly demote the failure to polite.
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
        // A counter rather than a bar: the wait has no progress to report, and
        // a bar that cannot honestly fill is worse than a number that climbs.
        <p className="backend-gate__elapsed">{elapsed}s</p>
      )}
    </motion.div>
  )
}
