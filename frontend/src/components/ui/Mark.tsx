import { motion } from 'motion/react'
import type { ComponentProps } from 'react'

/**
 * The product's spike glyph, drawn in ink so it follows the theme.
 *
 * A `motion.svg`, so callers can pass variants and transitions.
 */
export function Mark(props: ComponentProps<typeof motion.svg>) {
  return (
    <motion.svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" {...props}>
      <path d="M12 0 L14 10 L24 12 L14 14 L12 24 L10 14 L0 12 L10 10 Z" />
    </motion.svg>
  )
}
