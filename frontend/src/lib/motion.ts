// Shared Motion presets. Keep every transition here — never hand-roll configs
// inline per component (see .claude/skills/frontend-motion).
//
// Ivory Editorial moves on durations and named easings, not spring physics:
// reveals settle fast and long, and UI state changes stay quick and plain.
// The `springs` name stays so call sites keep working, but each entry is now a
// tween from the design system's motion table.
//
// Reveals are tuned per semantic:
//   - entrance ("soft arrival"): 400ms ease-out-quart — lands, then settles
//   - press / hover-class state changes: 150ms ease-out-quart
//   - color and crossfade tweens: 200ms / 400ms ease-out-quart
//
// Exits use ease-in-quart (the one "fast out" curve), which the library applies
// automatically whenever a transition carries `exit` preferences separately.

import type { Transition } from 'motion/react'

export const easings = {
  // Default for reveals, tab indicators, height changes.
  outQuart: [0.165, 0.84, 0.44, 1],
  // Color/background swaps only — the theme and section crossfades.
  inOutQuart: [0.77, 0, 0.175, 1],
  // Accordions and any height-driven open/close.
  standard: [0.4, 0, 0.2, 1],
  // Strips and carousels sliding.
  snap: [0.2, 0.8, 0.2, 1],
  // Exits only.
  inQuart: [0.895, 0.03, 0.685, 0.22],
} as const satisfies Record<string, readonly [number, number, number, number]>

// Timing scale (ms), straight from the design system's motion table.
export const timing = {
  fast: 0.15,
  base: 0.2,
  slide: 0.25,
  height: 0.4,
  reveal: 0.6,
  staggerStep: 0.04, // 40ms between words/rows in a staggered reveal
} as const

export const springs = {
  // Hover/prescale-state changes and icon entrances.
  standard: { duration: timing.base, ease: easings.outQuart } as Transition,

  // Press feedback — quick, no overshoot.
  press: { duration: timing.fast, ease: easings.outQuart } as Transition,

  // Rows and bubbles arriving: fade up out of nothing, then settle.
  card: { duration: timing.height, ease: easings.outQuart } as Transition,
}
