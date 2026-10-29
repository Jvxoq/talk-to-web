import type { ReactNode } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { PulseBlocks } from '../../../components/ui'
import { easeStandard, timing } from '../../../lib/motion'
import { useAuth } from '../hooks/useAuth'
import { AuthForm } from './AuthForm'

/**
 * Shows the app when signed in, the form otherwise.
 *
 * `restoring` renders neither, so a reload does not flash the sign-in form.
 */
export function AuthGate({ children }: { children: ReactNode }) {
  const { status } = useAuth()

  if (status === 'restoring') {
    return (
      <motion.div
        className="auth-restoring"
        initial={{ opacity: 0 }}
        // Delayed, so a fast refresh never flashes the indicator.
        animate={{ opacity: 1, transition: { delay: 0.4, duration: timing.standard } }}
        aria-live="polite"
        aria-busy
      >
        <PulseBlocks />
        <span className="sr-only">Loading</span>
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
        // A crossfade, since the whole screen changes. The exit is faster.
        transition={{ duration: timing.fast, ease: easeStandard }}
      >
        {status === 'signed-in' ? children : <AuthForm />}
      </motion.div>
    </AnimatePresence>
  )
}
