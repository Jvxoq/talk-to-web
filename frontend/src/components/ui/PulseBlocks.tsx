import './ui.css'

/**
 * Three ink blocks marching in step. The app's one "wait" indicator.
 *
 * No text and no percentage: neither wait it covers has honest progress to
 * report.
 */
export function PulseBlocks({ size = 'md' }: { size?: 'sm' | 'md' }) {
  return (
    <span
      className={size === 'sm' ? 'pulse-blocks pulse-blocks--sm' : 'pulse-blocks'}
      aria-hidden="true"
    >
      <i />
      <i />
      <i />
    </span>
  )
}
