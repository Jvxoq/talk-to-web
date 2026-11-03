import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { Moon, Sun } from 'lucide-react'
import { buttonClass } from '../../components/ui'
import { easings, timing } from '../../lib/motion'
import type { Theme } from './useTheme'

interface ThemeToggleProps {
  theme: Theme
  onToggle: () => void
}

export function ThemeToggle({ theme, onToggle }: ThemeToggleProps) {
  const reduce = useReducedMotion()

  return (
    <motion.button
      type="button"
      className={buttonClass('secondary', { icon: true })}
      onClick={onToggle}
      // The skill's contract for the theme toggle: an icon button, pressed when
      // the dark theme is live.
      aria-label="Dark mode"
      aria-pressed={theme === 'mono'}
      title="Dark mode"
      style={{ overflow: 'hidden' }}
    >
      <AnimatePresence mode="wait" initial={false}>
        <motion.span
          key={theme}
          style={{ display: 'flex' }}
          initial={reduce ? { opacity: 0 } : { rotate: -90, opacity: 0 }}
          animate={{ rotate: 0, opacity: 1 }}
          exit={reduce ? { opacity: 0 } : { rotate: 90, opacity: 0 }}
          transition={{ duration: timing.fast, ease: easings.outQuart }}
        >
          {theme === 'mono' ? (
            <Sun strokeWidth={1.5} aria-hidden="true" />
          ) : (
            <Moon strokeWidth={1.5} aria-hidden="true" />
          )}
        </motion.span>
      </AnimatePresence>
    </motion.button>
  )
}
