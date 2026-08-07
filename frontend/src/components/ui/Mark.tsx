import { motion } from 'motion/react'
import type { ComponentProps } from 'react'

/**
 * The product's spike glyph. A brand asset, not a design-system token — it
 * renders in ink and inverts with the theme like everything else.
 *
 * Built on `motion.svg` rather than `motion.create()` so callers can pass
 * variants and transitions straight through to a real animatable element.
 */
export function Mark(props: ComponentProps<typeof motion.svg>) {
  return (
    <motion.svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" {...props}>
      <path d="M12 0 L14 10 L24 12 L14 14 L12 24 L10 14 L0 12 L10 10 Z" />
    </motion.svg>
  )
}
