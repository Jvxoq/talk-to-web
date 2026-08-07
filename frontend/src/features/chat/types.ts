// Model identifiers are opaque strings the backend owns — fetched from
// `GET /models/` via `useModels`, never hardcoded here (see hooks/useModels.ts).
export type Model = string

/** One tool call's lifecycle, as reported by the `tool` SSE frame. */
export interface ToolActivity {
  name: string
  status: 'start' | 'ok' | 'failed'
  /** Present only on the `start` frame. */
  summary?: string
}

export interface Message {
  /** Stable across re-renders so AnimatePresence keys off identity, not order. */
  id: string
  role: 'user' | 'assistant'
  content: string
  error?: boolean
  /** Tool calls made while producing this (assistant) turn, keyed by name. */
  tools?: ToolActivity[]
}

export interface UploadedFile {
  name: string
  path: string
}
