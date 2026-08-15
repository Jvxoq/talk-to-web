import type { Usage } from './types'

/**
 * Narrows one untrusted `usage` SSE frame payload to the typed shape, at the
 * boundary — same pattern as the other frame-parsing helpers in `api.ts`. A
 * malformed payload (a future backend shape this build doesn't know yet)
 * returns `undefined` and the caller drops the frame, the same way an
 * unparsable SSE frame is dropped elsewhere in this file.
 */
export function parseUsage(raw: unknown): Usage | undefined {
  if (typeof raw !== 'object' || raw === null) return undefined
  const record = raw as Record<string, unknown>
  const { prompt_tokens, completion_tokens, elapsed_ms, model } = record

  if (
    typeof prompt_tokens !== 'number' ||
    typeof completion_tokens !== 'number' ||
    typeof elapsed_ms !== 'number' ||
    typeof model !== 'string'
  ) {
    return undefined
  }

  return {
    promptTokens: prompt_tokens,
    completionTokens: completion_tokens,
    elapsedMs: elapsed_ms,
    model,
  }
}

/** Plain token count for the chip — the two halves of a reply's spend added together. */
export function formatTokens(usage: Usage): string {
  return `${(usage.promptTokens + usage.completionTokens).toLocaleString()} tokens`
}

/**
 * Wall-clock time for the reply, in the coarsest unit that still reads
 * precisely: milliseconds below one second (where the digits matter), one
 * decimal place of seconds above it (where they stop being useful).
 */
export function formatDuration(usage: Usage): string {
  if (usage.elapsedMs < 1000) return `${Math.round(usage.elapsedMs)}ms`
  return `${(usage.elapsedMs / 1000).toFixed(1)}s`
}
