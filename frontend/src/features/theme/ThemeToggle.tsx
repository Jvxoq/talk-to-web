import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { Moon, Sun } from 'lucide-react'
import { buttonClass } from '../../components/ui'
import { springs } from '../../lib/motion'
import type { Theme } from './useTheme'

interface ThemeToggleProps {
  theme: Theme
  onToggle: () => void
}

export function ThemeToggle({ theme, onToggle }: ThemeToggleProps) {
  const reduce = useReducedMotion()
  const label = theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'

  return (
    <motion.button
      type="button"
      className={buttonClass('secondary', { icon: true })}
      onClick={onToggle}
      whileHover={{ x: -2, y: -2 }}
      whileTap={{ x: 0, y: 0 }}
      transition={springs.press}
      aria-label={label}
      title={label}
      style={{ overflow: 'hidden' }}
    >
      <AnimatePresence mode="wait" initial={false}>
        <motion.span
          key={theme}
          style={{ display: 'flex' }}
          initial={reduce ? { opacity: 0 } : { rotate: -90, opacity: 0 }}
          animate={{ rotate: 0, opacity: 1 }}
          exit={reduce ? { opacity: 0 } : { rotate: 90, opacity: 0 }}
          transition={{ duration: 0.18, ease: 'easeInOut' }}
        >
          {theme === 'dark' ? (
            <Sun strokeWidth={2} aria-hidden="true" />
          ) : (
            <Moon strokeWidth={2} aria-hidden="true" />
          )}
        </motion.span>
      </AnimatePresence>
    </motion.button>
  )
}
