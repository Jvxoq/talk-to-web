import { useEffect, useState } from 'react'

/** Whole seconds left until `until`, ticking once a second. `0` when there is no wait. */
export function useCountdown(until: number | null): number {
  const [secondsLeft, setSecondsLeft] = useState(() => remaining(until))

  useEffect(() => {
    setSecondsLeft(remaining(until))
    if (until === null) return

    const id = setInterval(() => {
      const left = remaining(until)
      setSecondsLeft(left)
      // Stop once the deadline has passed.
      if (left === 0) clearInterval(id)
    }, 1000)

    return () => clearInterval(id)
  }, [until])

  return secondsLeft
}

/**
 * Seconds left, from an absolute deadline. Background tabs throttle timers,
 * so a decrementing counter would drift.
 */
function remaining(until: number | null): number {
  if (until === null) return 0
  return Math.max(0, Math.ceil((until - Date.now()) / 1000))
}
