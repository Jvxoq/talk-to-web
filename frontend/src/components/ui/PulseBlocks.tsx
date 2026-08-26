import './ui.css'

/**
 * Three ink blocks marching in step. The app's one "wait" indicator.
 *
 * No text and no percentage: neither wait it covers has honest progress to
 * report.
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
