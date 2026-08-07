import { ApiError } from './http'

const CONVERSATIONS_URL = import.meta.env.VITE_CONVERSATIONS_URL ?? '/conversations/'
const STORAGE_KEY = 'conversationId'
const DEFAULT_TITLE = 'New conversation'

/** One persisted turn: the prompt and the response it produced. */
export interface MessageOut {
  id: number
  prompt_content: string
  response_content: string
  is_success?: boolean | null
  created_at?: string
}

export interface ConversationOut {
  id: number
  title: string
  model_type: string
  messages: MessageOut[]
}

/** The writable half of a turn — the server fills in ids and timestamps. */
export interface MessageCreate {
  prompt_content: string
  response_content: string
  is_success?: boolean
  status_code?: number
}

function endpoint(...segments: (string | number)[]): string {
  const base = CONVERSATIONS_URL.endsWith('/') ? CONVERSATIONS_URL : `${CONVERSATIONS_URL}/`
  return segments.length > 0 ? `${base}${segments.join('/')}` : base
}

async function requestJson(url: string, init?: RequestInit): Promise<unknown> {
  const response = await fetch(url, init)

  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null)
    const detail =
      typeof body === 'object' && body !== null && 'detail' in body
        ? String((body as { detail: unknown }).detail)
        : `Request failed with status ${response.status}`
    throw new ApiError(response.status, detail)
  }

  return response.json()
}

function isMessageOut(raw: unknown): raw is MessageOut {
  if (typeof raw !== 'object' || raw === null) return false
  const record = raw as Record<string, unknown>
  return (
    typeof record.id === 'number' &&
    typeof record.prompt_content === 'string' &&
    typeof record.response_content === 'string'
  )
}

/**
 * Narrows an untrusted response body to the fields we actually read.
 *
 * `messages` is optional on the wire — a create response has no turns yet, and
 * an endpoint that omits the relationship should hydrate to an empty log rather
 * than crash the app on load.
 */
function parseConversation(raw: unknown): ConversationOut {
  if (typeof raw !== 'object' || raw === null) {
    throw new ApiError(0, 'Conversation: expected an object')
  }
  const record = raw as Record<string, unknown>

  if (typeof record.id !== 'number') {
    throw new ApiError(0, 'Conversation: missing "id"')
  }

  return {
    id: record.id,
    title: typeof record.title === 'string' ? record.title : DEFAULT_TITLE,
    model_type: typeof record.model_type === 'string' ? record.model_type : '',
    messages: Array.isArray(record.messages) ? record.messages.filter(isMessageOut) : [],
  }
}

/**
 * Reads the conversation id pinned to this browser.
 *
 * Returns null when there is none — the caller creates one with
 * `createConversation` and pins the result via `storeConversationId`.
 * Storage access itself can throw (private browsing, blocked cookies), which
 * is treated the same as "no conversation yet" rather than a hard failure.
 */
export function getOrCreateConversationId(): number | null {
  let raw: string | null
  try {
    raw = localStorage.getItem(STORAGE_KEY)
  } catch {
    return null
  }

  if (raw === null) return null
  const id = Number(raw)
  return Number.isInteger(id) && id > 0 ? id : null
}

export function storeConversationId(id: number): void {
  try {
    localStorage.setItem(STORAGE_KEY, String(id))
  } catch {
    // Nothing to pin to; the session just won't survive a reload.
  }
}

export function clearConversationId(): void {
  try {
    localStorage.removeItem(STORAGE_KEY)
  } catch {
    // Already unreadable, so there is nothing to clear.
  }
}

export async function createConversation(
  modelType: string,
  title: string = DEFAULT_TITLE,
): Promise<ConversationOut> {
  const raw = await requestJson(endpoint(), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title, model_type: modelType }),
  })
  return parseConversation(raw)
}

export async function getConversation(id: number): Promise<ConversationOut> {
  return parseConversation(await requestJson(endpoint(id)))
}

export async function appendMessage(id: number, message: MessageCreate): Promise<void> {
  await requestJson(endpoint(id, 'messages'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(message),
  })
}

/**
 * Asks the server to drop the conversation as the page goes away.
 *
 * `sendBeacon` is the only request that reliably survives unload — a normal
 * fetch is cancelled with the document. It can only POST, hence the
 * `/delete` sub-path instead of a DELETE verb. Returns false when the browser
 * refuses to queue it (payload over the beacon budget, or no beacon support).
 */
export function deleteConversationBeacon(id: number): boolean {
  if (typeof navigator.sendBeacon !== 'function') return false

  const payload = new Blob([JSON.stringify({ id })], { type: 'application/json' })
  return navigator.sendBeacon(endpoint(id, 'delete'), payload)
}
