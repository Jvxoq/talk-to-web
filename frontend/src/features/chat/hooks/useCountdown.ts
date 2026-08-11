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
      // Stops itself rather than ticking against zero forever: the deadline is
      // absolute, so once it has passed there is nothing left to recompute.
      if (left === 0) clearInterval(id)
    }, 1000)

    return () => clearInterval(id)
  }, [until])

  return secondsLeft
}

/**
 * Derived from an absolute deadline, never counted down from a stored number.
 * A background tab throttles its timers to once a minute or stops them
 * entirely, so a decrementing counter would come back showing a wait that
 * expired minutes ago. Recomputing from the clock is always right on return.
 */
function remaining(until: number | null): number {
  if (until === null) return 0
  return Math.max(0, Math.ceil((until - Date.now()) / 1000))
}
