import './ui.css'

/**
 * Three ink blocks in step. The app's one "wait" indicator, with no progress shown.
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
