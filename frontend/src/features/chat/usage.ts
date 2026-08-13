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
  const { prompt_tokens, completion_tokens, cost_usd, model, priced } = record

  if (
    typeof prompt_tokens !== 'number' ||
    typeof completion_tokens !== 'number' ||
    typeof cost_usd !== 'number' ||
    typeof model !== 'string' ||
    typeof priced !== 'boolean'
  ) {
    return undefined
  }

  return {
    promptTokens: prompt_tokens,
    completionTokens: completion_tokens,
    costUsd: cost_usd,
    model,
    priced,
  }
}

/** Plain token count for the chip — the two halves of a reply's spend added together. */
export function formatTokens(usage: Usage): string {
  return `${(usage.promptTokens + usage.completionTokens).toLocaleString()} tokens`
}

/**
 * `priced: false` means no price was on file for that model, not that the
 * reply was free — so this never renders `$0.00` for an unpriced reply.
 *
 * A reply costs fractions of a cent, so `$0.00` at two decimal places would
 * be true and useless; six decimals is used below one cent, where the digits
 * are the information.
 */
export function formatCost(usage: Usage): string {
  if (!usage.priced) return 'cost unknown'
  const decimals = usage.costUsd > 0 && usage.costUsd < 0.01 ? 6 : 2
  return `$${usage.costUsd.toFixed(decimals)}`
}
