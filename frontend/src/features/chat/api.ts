import { ApiError, requireStringFields } from '../../lib/http'
import type { Model, ToolActivity, UploadedFile } from './types'

const API_URL = import.meta.env.VITE_API_URL ?? '/generate/text/'
const UPLOAD_URL = import.meta.env.VITE_UPLOAD_URL ?? '/upload/file/'
const MODELS_URL = import.meta.env.VITE_MODELS_URL ?? '/models/'

/** One parsed frame off the SSE stream. */
export type StreamEvent =
  | { type: 'delta'; text: string }
  | { type: 'tool'; activity: ToolActivity }
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
  done?: boolean
  error?: string
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
  const response = await fetch(API_URL, {
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
    throw new ApiError(response.status, `Request failed with status ${response.status}`)
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

  const response = await fetch(UPLOAD_URL, { method: 'POST', body: formData, signal })

  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null)
    const detail =
      typeof body === 'object' && body !== null && 'detail' in body
        ? String((body as { detail: unknown }).detail)
        : `Upload failed with status ${response.status}`
    throw new ApiError(response.status, detail)
  }

  const parsed = requireStringFields(await response.json(), ['file_path'], 'Upload')
  return { name: file.name, path: parsed.file_path }
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
  const response = await fetch(MODELS_URL, { signal })

  if (!response.ok) {
    throw new ApiError(response.status, `Request failed with status ${response.status}`)
  }

  return parseModelsResponse(await response.json())
}
