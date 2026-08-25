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
  /** Present once the `usage` frame lands, just before the assistant turn completes. */
  usage?: Usage
}

export interface UploadedFile {
  name: string
  path: string
}
