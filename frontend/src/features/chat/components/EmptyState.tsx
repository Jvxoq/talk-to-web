import { motion, useReducedMotion } from 'motion/react'
import { Mark } from '../../../components/ui/Mark'
import { springs } from '../../../lib/motion'

const rise = (reduce: boolean) => ({
  hidden: { opacity: 0, y: reduce ? 0 : 12 },
  show: { opacity: 1, y: 0, transition: springs.card },
})

export function EmptyState() {
  const reduce = useReducedMotion() ?? false

  return (
    <motion.div
      className="empty-state"
      initial="hidden"
      animate="show"
      variants={{
        hidden: {},
        show: { transition: { staggerChildren: 0.08, delayChildren: 0.15 } },
      }}
    >
      <Mark
        className="mark"
        variants={{
          hidden: { opacity: 0, scale: 0.6, rotate: reduce ? 0 : -90 },
          show: { opacity: 1, scale: 1, rotate: 0, transition: springs.card },
        }}
      />
      <motion.h2 variants={rise(reduce)}>
        Ask me anything <em>about the web.</em>
      </motion.h2>
      <motion.p variants={rise(reduce)}>
        Type, talk, or attach a PDF to get started.
      </motion.p>
    </motion.div>
  )
}
