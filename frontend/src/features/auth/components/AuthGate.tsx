import type { ReactNode } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { easeStandard, timing } from '../../../lib/motion'
import { useAuth } from '../hooks/useAuth'
import { AuthForm } from './AuthForm'

/**
 * Shows the app to whoever is signed in, and the form to everyone else.
 *
 * The third state matters as much as the other two: on a reload the access
 * token is gone and the refresh cookie has not been tried yet. Rendering the
 * form during that moment would flash a sign-in screen at someone who is
 * already signed in, so `restoring` renders neither.
 */
export function AuthGate({ children }: { children: ReactNode }) {
  const { status } = useAuth()

  if (status === 'restoring') {
    return (
      <motion.div
        className="auth-restoring"
        initial={{ opacity: 0 }}
        // Held back deliberately: the refresh usually answers in well under
        // this, and a spinner that appears and vanishes reads as a glitch.
        animate={{ opacity: 1, transition: { delay: 0.4, duration: timing.standard } }}
        aria-live="polite"
      >
        <span>Restoring your session…</span>
      </motion.div>
    )
  }

  return (
    <AnimatePresence mode="wait" initial={false}>
      <motion.div
        key={status}
        className="auth-gate"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        // A crossfade rather than a slide: signing in replaces the whole screen,
        // and there is no spatial relationship between the two to preserve.
        // The exit is the faster half, so nothing waits on what is leaving.
        transition={{ duration: timing.fast, ease: easeStandard }}
      >
        {status === 'signed-in' ? children : <AuthForm />}
      </motion.div>
    </AnimatePresence>
  )
}
