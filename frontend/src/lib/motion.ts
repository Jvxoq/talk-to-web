// Shared Motion presets. Keep every spring here — never hand-roll configs
// inline per component (see .claude/skills/frontend-motion).
//
// The brand voice is calm, considered, physical: springs are critically or
// slightly over-damped, timing sits on the slower end of smooth. Use spring
// physics for anything involving position / scale / size; reserve duration +
// easeStandard tweens for pure opacity / colour fades.

import type { Transition } from 'motion/react'

export const springs = {
  // Default for entrances, layout animations, most UI motion.
  standard: { type: 'spring', stiffness: 260, damping: 30, mass: 1 },

  // Buttons, icon-buttons, tab switches — snappier, no overshoot.
  press: { type: 'spring', stiffness: 400, damping: 35, mass: 0.8 },

  // Cards / bubbles entering & leaving.
  card: { type: 'spring', stiffness: 220, damping: 26, mass: 1 },
} satisfies Record<string, Transition>

// ease-out, for opacity / colour-only tweens where a spring is overkill.
export const easeStandard = [0.22, 1, 0.36, 1] as const

// Timing scale (ms). Nothing not backed by a loading state exceeds ~450ms.
export const timing = {
  instant: 0.1,
  fast: 0.18,
  standard: 0.26,
  entrance: 0.32,
  staggerStep: 0.06,
} as const
