import { motion, useReducedMotion } from 'motion/react'

/**
 * The "thinking" state before the first token lands — three ink squares that
 * rise and fade in sequence. Falls back to static squares under reduced motion.
 */
export function TypingDots() {
  const reduce = useReducedMotion()

  return (
    <span className="typing" aria-label="Assistant is thinking">
      {[0, 1, 2].map((i) => (
        <motion.span
          key={i}
          animate={reduce ? undefined : { opacity: [0.3, 1, 0.3], y: [0, -3, 0] }}
          transition={{ duration: 1, repeat: Infinity, ease: 'easeInOut', delay: i * 0.15 }}
        />
      ))}
    </span>
  )
}
