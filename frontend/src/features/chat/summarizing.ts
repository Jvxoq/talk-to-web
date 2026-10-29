import type { Summarizing } from './types'

/**
 * Narrows a `summarizing` SSE payload, or returns `undefined` to drop it.
 *
 * `tokens_after` stays `undefined` on `start`, never 0.
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

/** `12480` -> `12.5k`. */
export function formatTokenCount(tokens: number): string {
  if (tokens < 1000) return `${tokens}`
  return `${(tokens / 1000).toFixed(1)}k`
}

/**
 * The chip's text. States the status in words, for screen readers.
 */
export function summarizingText(summarizing: Summarizing): string {
  if (summarizing.status === 'start') return 'Condensing context…'
  if (summarizing.tokensAfter === undefined) return 'Context condensed'
  return `Context condensed ${formatTokenCount(summarizing.tokensBefore)} → ${formatTokenCount(
    summarizing.tokensAfter,
  )}`
}
