import type { Summarizing } from './types'

/**
 * Narrows one untrusted `summarizing` SSE frame payload, at the boundary —
 * same pattern as `parseUsage`. A payload this build doesn't recognise returns
 * `undefined` and the caller drops the frame rather than tearing the stream
 * down.
 *
 * `tokens_after` is absent on the `start` frame: the shortened history does
 * not exist yet, so there is no number to send. That stays `undefined` here
 * rather than becoming 0, which would read as "condensed to nothing".
 */
export function parseSummarizing(raw: unknown): Summarizing | undefined {
  if (typeof raw !== 'object' || raw === null) return undefined
  const record = raw as Record<string, unknown>
  const { status, tokens_before, tokens_after } = record

  if (status !== 'start' && status !== 'done') return undefined
  if (typeof tokens_before !== 'number') return undefined
  if (tokens_after !== undefined && typeof tokens_after !== 'number') return undefined

  return { status, tokensBefore: tokens_before, tokensAfter: tokens_after }
}

/** `12480` -> `12.5k`. Chip-sized: the magnitude is the point, not the digits. */
export function formatTokenCount(tokens: number): string {
  if (tokens < 1000) return `${tokens}`
  return `${(tokens / 1000).toFixed(1)}k`
}

/**
 * The chip's text. Carries the state in words, not colour, so it reads the
 * same to a screen reader as it does on screen.
 */
export function summarizingText(summarizing: Summarizing): string {
  if (summarizing.status === 'start') return 'Condensing context…'
  if (summarizing.tokensAfter === undefined) return 'Context condensed'
  return `Context condensed ${formatTokenCount(summarizing.tokensBefore)} → ${formatTokenCount(
    summarizing.tokensAfter,
  )}`
}
