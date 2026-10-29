import type { Usage } from './types'

/**
 * Narrows a `usage` SSE payload, or returns `undefined` to drop it.
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
 * The reply time: milliseconds under one second, else seconds with one decimal.
 */
export function formatDuration(usage: Usage): string {
  if (usage.elapsedMs < 1000) return `${Math.round(usage.elapsedMs)}ms`
  return `${(usage.elapsedMs / 1000).toFixed(1)}s`
}
