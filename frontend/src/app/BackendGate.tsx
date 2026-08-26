import { useCallback, useEffect, useState, type ReactNode } from 'react'
import { motion } from 'motion/react'
import { buttonClass, PulseBlocks } from '../components/ui'
import { easeStandard, springs, timing } from '../lib/motion'
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
    setStatus('checking')
    setRevealed(false)
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
      transition={{ duration: timing.standard, ease: easeStandard }}
      // `role="alert"` is already assertive, and a stated `aria-live` would
      // demote the failure to polite.
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
