import { motion, useReducedMotion } from 'motion/react'
import { springs, timing } from '../../../lib/motion'

interface GhostRow {
  /** Stable across renders. The list is static, but never key on index. */
  id: string
  role: 'user' | 'assistant'
  lines: readonly string[]
}

/**
 * The shape of a loaded transcript: one settled assistant turn above, then the
 * exchange being asked about. Ending on an assistant bubble is what makes it
 * read as history rather than as a form, and the short final bar suggests the
 * reply was still being written.
 *
 * Widths only. Nothing here claims to know how long anything was — the point
 * is that the real bubbles land into geometry that already exists.
 */
const GHOST_ROWS: readonly GhostRow[] = [
  { id: 'ghost-a', role: 'assistant', lines: ['is-short', 'is-mid'] },
  { id: 'ghost-u', role: 'user', lines: ['is-mid'] },
  { id: 'ghost-b', role: 'assistant', lines: ['is-full', 'is-long', 'is-mid'] },
]

const rise = (reduce: boolean) => ({
  hidden: { opacity: 0, y: reduce ? 0 : 12 },
  show: { opacity: 1, y: 0, transition: springs.card },
})

/**
 * Holds the transcript's place while a conversation is fetched.
 *
 * The rows are the real ones — same `.message`, `.message-role` and `.bubble`
 * classes the transcript uses — with the text swapped for bars. Only the
 * placeholder fill is added here, so the two cannot drift apart: change a
 * bubble's padding and the skeleton already moves with it.
 *
 * Every row is `aria-hidden`. The wait is announced once by the status below,
 * and the bars are decoration, so a screen reader has nothing to gain from
 * walking them.
 */
export function ConversationSkeleton() {
  const reduce = useReducedMotion() ?? false

  return (
    <motion.div
      className="skeleton"
      initial="hidden"
      animate="show"
      variants={{
        hidden: {},
        show: { transition: { staggerChildren: timing.staggerStep, delayChildren: 0.1 } },
      }}
    >
      <span className="sr-only" role="status">
        Loading conversation
      </span>

      {GHOST_ROWS.map((row) => (
        <motion.div key={row.id} className={`message ${row.role}`} variants={rise(reduce)} aria-hidden="true">
          <span className="message-role skeleton-role" />
          <div className="bubble skeleton-bubble">
            {row.lines.map((width, i) => (
              <span key={`${row.id}-${i}`} className={`skeleton-line ${width}`} />
            ))}
          </div>
        </motion.div>
      ))}
    </motion.div>
  )
}