import { ApiError, requireStringFields } from '../../lib/http'
import { authorizedFetch } from '../../lib/session'
import { parseUsage } from './usage'
import type { DocumentSummary, Model, ToolActivity, UploadedFile, Usage } from './types'

const API_URL = import.meta.env.VITE_API_URL ?? '/generate/text/'
const UPLOAD_URL = import.meta.env.VITE_UPLOAD_URL ?? '/upload/file/'
const INGEST_URL_URL = import.meta.env.VITE_INGEST_URL_URL ?? '/upload/url/'
const DOCUMENTS_URL = import.meta.env.VITE_DOCUMENTS_URL ?? '/documents/'
const MODELS_URL = import.meta.env.VITE_MODELS_URL ?? '/models/'

/** One parsed frame off the SSE stream. */
export type StreamEvent =
  | { type: 'delta'; text: string }
  | { type: 'tool'; activity: ToolActivity }
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
  signal?: AbortSignal,
): Promise<UploadedFile> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await authorizedFetch(UPLOAD_URL, {
    method: 'POST',
    body: formData,
    signal,
  })

  if (!response.ok) {
    const failure = await failureOf(response, 'Upload failed')
    throw new ApiError(response.status, failure.detail, failure.retryAfterSeconds)
  }

  const parsed = requireStringFields(await response.json(), ['file_path'], 'Upload')
  return { name: file.name, path: parsed.file_path }
}

/** Fetches a URL server-side and indexes its full text, the same way an upload is. */
export async function ingestUrl(url: string, signal?: AbortSignal): Promise<UploadedFile> {
  const response = await authorizedFetch(INGEST_URL_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
    signal,
  })

  if (!response.ok) {
    const failure = await failureOf(response, 'Could not fetch that URL')
    throw new ApiError(response.status, failure.detail, failure.retryAfterSeconds)
  }

  const parsed = requireStringFields(await response.json(), ['file_path'], 'Ingest')
  return { name: url, path: parsed.file_path }
}

function documentEndpoint(...segments: (string | number)[]): string {
  const base = DOCUMENTS_URL.endsWith('/') ? DOCUMENTS_URL : `${DOCUMENTS_URL}/`
  return segments.length > 0 ? `${base}${segments.join('/')}` : base
}

/** Narrows one untrusted `/documents/` list item to the fields the panel reads. */
function parseDocument(raw: unknown): DocumentSummary {
  if (typeof raw !== 'object' || raw === null) {
    throw new ApiError(0, 'Document: expected an object')
  }
  const record = raw as Record<string, unknown>

  if (typeof record.id !== 'number') {
    throw new ApiError(0, 'Document: missing "id"')
  }
  if (typeof record.name !== 'string') {
    throw new ApiError(0, 'Document: missing "name"')
  }

  return {
    id: record.id,
    name: record.name,
    chunksIndexed: typeof record.chunks_indexed === 'number' ? record.chunks_indexed : 0,
  }
}

/** Every document this account has uploaded, for the document manager panel. */
export async function fetchDocuments(signal?: AbortSignal): Promise<DocumentSummary[]> {
  const response = await authorizedFetch(documentEndpoint(), { signal })

  if (!response.ok) {
    throw new ApiError(response.status, `Request failed with status ${response.status}`)
  }

  const raw: unknown = await response.json()
  if (!Array.isArray(raw)) {
    throw new ApiError(0, 'Documents: expected an array')
  }
  return raw.map(parseDocument)
}

/**
 * Deletes a document: its vectors, its stored file, and its row.
 *
 * A POST, not a DELETE, for the same CORS reason as `deleteConversation` in
 * `lib/conversation.ts` — this app allows only GET, POST and OPTIONS
 * cross-origin.
 */
export async function deleteDocument(id: number): Promise<void> {
  const response = await authorizedFetch(documentEndpoint(id, 'delete'), { method: 'POST' })

  // 404 means it is already gone, which is the outcome the caller wanted.
  if (!response.ok && response.status !== 404) {
    throw new ApiError(response.status, `Could not delete document ${id}`)
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
