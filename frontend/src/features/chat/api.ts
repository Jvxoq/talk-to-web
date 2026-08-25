import { ApiError, requireStringFields } from '../../lib/http'
import { authorizedFetch } from '../../lib/session'
import { parseSummarizing } from './summarizing'
import { parseUsage } from './usage'
import type { Model, Summarizing, ToolActivity, UploadedFile, Usage } from './types'

const API_URL = import.meta.env.VITE_API_URL ?? '/generate/text/'
const UPLOAD_URL = import.meta.env.VITE_UPLOAD_URL ?? '/upload/file/'
const DOCUMENTS_URL = import.meta.env.VITE_DOCUMENTS_URL ?? '/documents/'
const MODELS_URL = import.meta.env.VITE_MODELS_URL ?? '/models/'

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
 * The backend's account of a failure: its own sentence, and — on a 429 — the
 * number of seconds it wants us to wait.
 *
 * The wait is read from the body rather than the `Retry-After` header because a
 * browser cannot see that header unless the server exposes it, and this app is
 * not always same-origin with its API. Narrowed here, at the boundary, then
 * trusted downstream.
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
 * Frames carry a JSON object rather than bare text, because a token can contain
 * the newlines that markdown structure is made of and a raw newline would end
 * the frame early. Failures are reported in-band as `{ error }` — so an error
 * can arrive after a 200, and the happy path alone isn't enough to know it
 * worked.
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
    // The body is read for its `detail`, not discarded: a refusal that arrives
    // before the stream opens - a spent rate-limit budget is the one that
    // actually happens - carries the only sentence that tells the user how long
    // to wait. "Request failed with status 429" tells them nothing.
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
          // A frame we can't read is one token's worth of text; dropping it
          // beats tearing down a stream that is otherwise fine.
          continue
        }

        // Order matters: error and done are terminal, tool must land before
        // delta so a frame carrying both `tool` and no `delta` doesn't fall
        // through silently. Unknown frames keep falling through harmlessly.
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
          // A frame this build doesn't yet know the shape of is dropped, the
          // same way an unparsable frame is dropped above — not a reason to
          // tear down an otherwise-working stream.
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
  // The thread the file is attached to. A document belongs to one conversation
  // and is only searchable from that one, so there is no upload without it.
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
  const parsed = requireStringFields(body, ['file_path'], 'Upload')
  const id = (body as Record<string, unknown>).document_id
  if (typeof id !== 'number') {
    throw new ApiError(0, 'Upload: missing "document_id"')
  }
  return { name: file.name, path: parsed.file_path, id }
}

/**
 * Detaches a document: its passages, its stored file and its row all go.
 *
 * A POST rather than a DELETE for the same reason conversation deletion is one:
 * this app's CORS policy allows GET, POST and OPTIONS across origins, nothing
 * else. A 404 means it is already gone, which is the outcome the caller wanted.
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
