import { useCallback, useState, type FormEvent } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { Mark } from '../../../components/ui'
import { buttonClass } from '../../../components/ui'
import { easeStandard, springs, timing } from '../../../lib/motion'
import { messageFrom } from '../../../lib/http'
import { useAuth } from '../hooks/useAuth'
import type { AuthMode } from '../types'

const COPY: Record<AuthMode, { title: string; action: string; switchTo: string }> = {
  login: { title: 'Sign in', action: 'Sign in', switchTo: 'Need an account? Sign up' },
  register: { title: 'Sign up', action: 'Create account', switchTo: 'Already have an account?' },
}

/** Matches the backend's `MIN_PASSWORD_LENGTH`, purely to say so before submitting. */
const MIN_PASSWORD_LENGTH = 12

export function AuthForm() {
  const { submit } = useAuth()
  const reduce = useReducedMotion() ?? false

  const [mode, setMode] = useState<AuthMode>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const copy = COPY[mode]
  const canSubmit = !busy && email.trim().length > 0 && password.length > 0

  const switchMode = useCallback(() => {
    setMode((prev) => (prev === 'login' ? 'register' : 'login'))
    // The password is cleared, the address is not: switching after a failed
    // sign-in almost always means "same person, wrong form".
    setPassword('')
    setError(null)
  }, [])

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!canSubmit) return

    setBusy(true)
    setError(null)
    try {
      await submit(mode, { email: email.trim(), password })
    } catch (err) {
      setError(messageFrom(err, 'Something went wrong. Please try again.'))
    } finally {
      setBusy(false)
    }
  }

  return (
    <motion.div
      className="auth"
      initial="hidden"
      animate="show"
      variants={{
        hidden: {},
        show: { transition: { staggerChildren: timing.staggerStep, delayChildren: 0.1 } },
      }}
    >
      <motion.div
        variants={{
          hidden: { opacity: 0, scale: 0.6, rotate: reduce ? 0 : -90 },
          show: { opacity: 1, scale: 1, rotate: 0, transition: springs.card },
        }}
      >
        <Mark className="mark" />
      </motion.div>

      <motion.h1
        variants={{
          hidden: { opacity: 0, y: reduce ? 0 : 12 },
          show: { opacity: 1, y: 0, transition: springs.card },
        }}
      >
        {copy.title}
      </motion.h1>

      <motion.form
        className="auth__form"
        onSubmit={handleSubmit}
        variants={{
          hidden: { opacity: 0, y: reduce ? 0 : 12 },
          show: { opacity: 1, y: 0, transition: springs.card },
        }}
      >
        <label className="auth__field">
          <span>Email</span>
          <input
            type="email"
            name="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            autoComplete="email"
            required
            disabled={busy}
          />
        </label>

        <label className="auth__field">
          <span>Password</span>
          <input
            type="password"
            name="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            // Tells the password manager which form this is, so it offers to
            // save a new password on sign-up and fill the saved one on sign-in.
            autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
            minLength={mode === 'register' ? MIN_PASSWORD_LENGTH : undefined}
            required
            disabled={busy}
          />
          {mode === 'register' && (
            <small>At least {MIN_PASSWORD_LENGTH} characters.</small>
          )}
        </label>

        <AnimatePresence initial={false}>
          {error !== null && (
            <motion.p
              className="auth__error"
              role="alert"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: timing.standard, ease: easeStandard }}
              style={{ overflow: 'hidden' }}
            >
              {error}
            </motion.p>
          )}
        </AnimatePresence>

        <motion.button
          type="submit"
          className={buttonClass('primary')}
          disabled={!canSubmit}
          whileHover={canSubmit ? { x: -2, y: -2 } : undefined}
          whileTap={canSubmit ? { x: 0, y: 0 } : undefined}
          transition={springs.press}
        >
          {busy ? 'Working…' : copy.action}
        </motion.button>
      </motion.form>

      <motion.button
        type="button"
        className="auth__switch"
        onClick={switchMode}
        disabled={busy}
        variants={{
          hidden: { opacity: 0, y: reduce ? 0 : 12 },
          show: { opacity: 1, y: 0, transition: springs.card },
        }}
      >
        {copy.switchTo}
      </motion.button>
    </motion.div>
  )
}
