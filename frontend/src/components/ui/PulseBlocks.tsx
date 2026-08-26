import './ui.css'

/**
 * Three ink blocks marching in step. The app's one "wait" indicator.
 *
 * No text and no percentage: it says "wait" without naming the mechanism, and
 * neither wait it covers — restoring a session, waking a sleeping backend —
 * has honest progress to report. Shared rather than copied because both of
 * those screens want the same shape, and a second copy is how the two drift.
 */
export function PulseBlocks() {
  return (
    <span className="pulse-blocks" aria-hidden="true">
      <i />
      <i />
      <i />
    </span>
  )
}
