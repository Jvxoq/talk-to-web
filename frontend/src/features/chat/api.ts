import { ApiError } from '../../lib/http'
import { authorizedFetch } from '../../lib/session'
import { parseSummarizing } from './summarizing'
import { parseUsage } from './usage'
import type { Model, Summarizing, ToolActivity, UploadedFile, Usage } from './types'

const API_URL = import.meta.env.VITE_API_URL ?? '/generate/text/'
const UPLOAD_URL = import.meta.env.VITE_UPLOAD_URL ?? '/upload/file/'
const DOCUMENTS_URL = import.meta.env.VITE_DOCUMENTS_URL ?? '/documents/'
const MODELS_URL = import.meta.env.VITE_MODELS_URL ?? '/models/'
const CONVERSATIONS_URL = import.meta.env.VITE_CONVERSATIONS_URL ?? '/conversations/'

/** One parsed frame off the SSE stream. */
export type StreamEvent =
  | { type: 'delta'; text: string }
  | { type: 'tool'; activity: ToolActivity }
  | { type: 'summarizing'; summarizing: Summarizing }
  | { type: 'usage'; usage: Usage }
  | { type: 'error'; message: string }

interface StreamRequest {
  model: Model
  userInput: string
  /** `null` until the conversation has been created server-side. */
  conversationId: number | null
  signal?: AbortSignal
}

/** The JSON body of one `data:` frame. Exactly one key is set per frame. */
interface StreamFrame {
  delta?: string
  tool?: ToolActivity
  /** Raw and unnarrowed here — `parseSummarizing` does the boundary check below. */
  summarizing?: unknown
  /** Raw and unnarrowed here — `parseUsage` does the boundary check below. */
  usage?: unknown
  done?: boolean
  error?: string
}

/**
 * The backend's failure message and, on a 429, the seconds to wait.
 *
 * The wait comes from the body, since cross-origin JS may not read `Retry-After`.
 */
async function failureOf(
  response: Response,
  fallback = 'Request failed',
): Promise<{ detail: string; retryAfterSeconds?: number }> {
  const body: unknown = await response.json().catch(() => null)
  const record = typeof body === 'object' && body !== null ? (body as Record<string, unknown>) : {}

  const detail =
    typeof record.detail === 'string'
      ? record.detail
      : `${fallback} with status ${response.status}`
  const retry = record.retry_after_seconds

  return {
    detail,
    retryAfterSeconds: typeof retry === 'number' && retry > 0 ? retry : undefined,
  }
}

/**
 * Opens the chat stream and yields one event per SSE frame.
 *
 * Each frame is JSON, so newlines in tokens are safe. Errors arrive in-band as
 * `{ error }`, even after a 200.
 */
export async function* streamChat({
  model,
  userInput,
  conversationId,
  signal,
}: StreamRequest): AsyncGenerator<StreamEvent> {
  const response = await authorizedFetch(API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      user_input: userInput,
      temperature: 0,
      conversation_id: conversationId,
    }),
    signal,
  })

  if (!response.ok || !response.body) {
    // Read the body: a 429's `detail` tells the user how long to wait.
    const failure = await failureOf(response)
    throw new ApiError(response.status, failure.detail, failure.retryAfterSeconds)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) return

      buffer += decoder.decode(value, { stream: true })
      const frames = buffer.split('\n\n')
      buffer = frames.pop() ?? ''

      for (const frame of frames) {
        if (!frame.startsWith('data: ')) continue

        let parsed: StreamFrame
        try {
          parsed = JSON.parse(frame.slice('data: '.length)) as StreamFrame
        } catch {
          // Drop an unreadable frame rather than end the stream.
          continue
        }

        // Order matters: error and done end the stream. Unknown frames are ignored.
        if (parsed.error !== undefined) {
          yield { type: 'error', message: parsed.error }
          return
        }
        if (parsed.done) return
        if (parsed.tool !== undefined) {
          yield { type: 'tool', activity: parsed.tool }
        }
        if (parsed.summarizing !== undefined) {
          const summarizing = parseSummarizing(parsed.summarizing)
          if (summarizing) yield { type: 'summarizing', summarizing }
        }
        if (parsed.usage !== undefined) {
          // A malformed usage frame is dropped, not fatal.
          const usage = parseUsage(parsed.usage)
          if (usage) yield { type: 'usage', usage }
        }
        if (parsed.delta) yield { type: 'delta', text: parsed.delta }
      }
    }
  } finally {
    // Releasing the lock lets an aborted fetch tear the body down cleanly.
    reader.releaseLock()
  }
}

export async function uploadPdf(
  file: File,
  conversationId: number,
  signal?: AbortSignal,
): Promise<UploadedFile> {
  const formData = new FormData()
  formData.append('file', file)
  // Every upload belongs to one conversation.
  formData.append('conversation_id', String(conversationId))

  const response = await authorizedFetch(UPLOAD_URL, {
    method: 'POST',
    body: formData,
    signal,
  })

  if (!response.ok) {
    const failure = await failureOf(response, 'Upload failed')
    throw new ApiError(response.status, failure.detail, failure.retryAfterSeconds)
  }

  const body: unknown = await response.json()
  // Only `document_id` is needed. `file_path` is ignored.
  const id = (body as Record<string, unknown>).document_id
  if (typeof id !== 'number') {
    throw new ApiError(0, 'Upload: missing "document_id"')
  }
  return { name: file.name, id }
}

/**
 * The document attached to a conversation, or `null`.
 *
 * Read when a thread opens, since the attachment outlives the tab.
 */
export async function getAttachedDocument(conversationId: number): Promise<UploadedFile | null> {
  const base = CONVERSATIONS_URL.endsWith('/') ? CONVERSATIONS_URL : `${CONVERSATIONS_URL}/`
  const response = await authorizedFetch(`${base}${conversationId}/document`)

  if (!response.ok) {
    throw new ApiError(response.status, `Could not load the attachment for conversation ${conversationId}`)
  }

  const body: unknown = await response.json()
  // `null` means no document, which is normal.
  if (body === null) return null

  // Narrowed here, then trusted. The chip needs only the name and id.
  if (typeof body !== 'object') {
    throw new ApiError(0, 'Document: expected an object')
  }
  const record = body as Record<string, unknown>
  const id = record.id
  const name = record.name
  if (typeof id !== 'number' || typeof name !== 'string') {
    throw new ApiError(0, 'Document: missing "id" or "name"')
  }
  return { id, name }
}

/**
 * Deletes a document: passages, stored file and row.
 *
 * POST, because CORS allows only GET, POST and OPTIONS. A 404 means already gone.
 */
export async function deleteDocument(documentId: number): Promise<void> {
  const base = DOCUMENTS_URL.endsWith('/') ? DOCUMENTS_URL : `${DOCUMENTS_URL}/`
  const response = await authorizedFetch(`${base}${documentId}/delete`, { method: 'POST' })

  if (!response.ok && response.status !== 404) {
    throw new ApiError(response.status, `Could not remove document ${documentId}`)
  }
}

export interface ModelsResponse {
  models: string[]
  default: string
}

/** Narrows an untrusted `/models/` response to the two fields we read. */
function parseModelsResponse(raw: unknown): ModelsResponse {
  if (typeof raw !== 'object' || raw === null) {
    throw new ApiError(0, 'Models: expected an object')
  }
  const record = raw as Record<string, unknown>

  if (!Array.isArray(record.models) || !record.models.every((m) => typeof m === 'string')) {
    throw new ApiError(0, 'Models: missing "models"')
  }
  if (typeof record.default !== 'string') {
    throw new ApiError(0, 'Models: missing "default"')
  }

  return { models: record.models, default: record.default }
}

export async function fetchModels(signal?: AbortSignal): Promise<ModelsResponse> {
  const response = await authorizedFetch(MODELS_URL, { signal })

  if (!response.ok) {
    throw new ApiError(response.status, `Request failed with status ${response.status}`)
  }

  return parseModelsResponse(await response.json())
}
