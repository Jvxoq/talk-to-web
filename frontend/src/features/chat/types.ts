// Model identifiers are opaque strings the backend owns — fetched from
// `GET /models/` via `useModels`, never hardcoded here (see hooks/useModels.ts).
export type Model = string

/** Where a piece of grounding came from — a document name, or a page with a link. */
export interface Source {
  label: string
  /** Absent for a passage retrieved from the user's own upload — nothing to link to. */
  url?: string
}

/** One tool call's lifecycle, as reported by the `tool` SSE frame. */
export interface ToolActivity {
  name: string
  status: 'start' | 'ok' | 'failed'
  /** Present only on the `start` frame. */
  summary?: string
  /** Present only on a successful `finished` frame that found something to cite. */
  sources?: Source[]
}

/**
 * The agent shortening a long thread so it still fits the model's budget.
 *
 * Reported by the `summarizing` SSE frame. It is a whole model call the user
 * waits through with no text arriving, which is why it is on screen at all.
 */
export interface Summarizing {
  status: 'start' | 'done'
  /** Size of the thread that triggered it, in tokens. */
  tokensBefore: number
  /** Size after condensing. Absent on the `start` frame — not known yet. */
  tokensAfter?: number
}

/** What one reply spent, as reported by the `usage` SSE frame. */
export interface Usage {
  promptTokens: number
  completionTokens: number
  /** Wall-clock time for the whole reply, in milliseconds. */
  elapsedMs: number
  model: string
}

export interface Message {
  /** Stable across re-renders so AnimatePresence keys off identity, not order. */
  id: string
  role: 'user' | 'assistant'
  content: string
  error?: boolean
  /** Tool calls made while producing this (assistant) turn, keyed by name. */
  tools?: ToolActivity[]
  /** Set while (and after) this turn's history had to be condensed. */
  summarizing?: Summarizing
  /** Present once the `usage` frame lands, just before the assistant turn completes. */
  usage?: Usage
}

export interface UploadedFile {
  name: string
  path: string
  /** The row this upload created. Removing the chip deletes it by this id. */
  id: number
}
