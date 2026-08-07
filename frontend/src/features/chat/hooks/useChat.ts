import { useCallback, useEffect, useRef, useState } from 'react'
import {
  appendMessage,
  clearConversationId,
  createConversation,
  deleteConversationBeacon,
  getConversation,
  getOrCreateConversationId,
  storeConversationId,
  type MessageCreate,
  type MessageOut,
} from '../../../lib/conversation'
import { ApiError, isAbort } from '../../../lib/http'
import { streamChat } from '../api'
import type { Message, Model, ToolActivity } from '../types'

const GENERIC_FAILURE = 'Something went wrong. Please try again.'

/**
 * A stored turn is one prompt/response row; the transcript renders it as the
 * two bubbles the user originally saw. Ids are derived from the row so they
 * stay stable across re-renders, the way `crypto.randomUUID` ones do live.
 */
function toMessages(turns: MessageOut[]): Message[] {
  return turns.flatMap((turn): Message[] => [
    { id: `${turn.id}-user`, role: 'user', content: turn.prompt_content },
    {
      id: `${turn.id}-assistant`,
      role: 'assistant',
      content: turn.response_content,
      error: turn.is_success === false,
    },
  ])
}

/**
 * Picks up the pinned conversation, or opens a fresh one.
 *
 * A stored id routinely goes stale: the unload beacon deletes the row as the
 * page goes away, so anything that survives is from a page-life that never got
 * to run its beacon. A 404 there is expected, not an error.
 */
async function resume(model: Model): Promise<{ id: number; messages: Message[] }> {
  const storedId = getOrCreateConversationId()

  if (storedId !== null) {
    try {
      const conversation = await getConversation(storedId)
      return { id: conversation.id, messages: toMessages(conversation.messages) }
    } catch (error) {
      if (!(error instanceof ApiError) || error.status !== 404) throw error
      clearConversationId()
    }
  }

  const created = await createConversation(model)
  storeConversationId(created.id)
  return { id: created.id, messages: [] }
}

/**
 * Owns the transcript and the in-flight stream.
 *
 * The assistant's turn is appended empty and filled in place as deltas land, so
 * the list renders a pending bubble without a second piece of state to keep in
 * sync with `messages`.
 *
 * Turns are mirrored to the server as they complete, but that mirror is
 * best-effort: what is on screen stays authoritative for this page-life, and
 * the conversation is dropped again on unload.
 */
export function useChat(model: Model) {
  const [messages, setMessages] = useState<Message[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const conversationIdRef = useRef<number | null>(null)
  // Held as a promise so a turn sent before bootstrap settles still gets
  // persisted, rather than being dropped against a null id.
  const bootstrapRef = useRef<Promise<number | null> | null>(null)
  // Read at bootstrap only, via a ref, so switching models mid-session doesn't
  // re-run the effect and open a second conversation.
  const modelRef = useRef(model)
  useEffect(() => {
    modelRef.current = model
  }, [model])

  useEffect(() => {
    // The ref survives StrictMode's remount, so exactly one conversation is
    // opened in development as well as production.
    if (bootstrapRef.current) return

    bootstrapRef.current = resume(modelRef.current)
      .then(({ id, messages: restored }) => {
        conversationIdRef.current = id
        if (restored.length > 0) setMessages(restored)
        return id
      })
      .catch(() => null)
  }, [])

  useEffect(() => {
    const handleExit = (event: Event | PageTransitionEvent) => {
      // A bfcache'd page can be restored, so its conversation has to outlive it.
      if ('persisted' in event && event.persisted) return

      const id = conversationIdRef.current
      if (id === null) return

      // Cleared first: browsers fire both events on an ordinary unload, and the
      // second beacon would hit an already-deleted row.
      conversationIdRef.current = null
      deleteConversationBeacon(id)
      clearConversationId()
    }

    window.addEventListener('pagehide', handleExit)
    window.addEventListener('beforeunload', handleExit)
    return () => {
      window.removeEventListener('pagehide', handleExit)
      window.removeEventListener('beforeunload', handleExit)
    }
  }, [])

  const persist = useCallback(async (message: MessageCreate) => {
    const id = conversationIdRef.current ?? (await bootstrapRef.current)
    if (id == null) return

    try {
      await appendMessage(id, message)
    } catch {
      // Deliberately swallowed: a failed write is invisible to the user, and
      // surfacing it would break up a conversation that otherwise worked.
    }
  }, [])

  const failLastMessage = useCallback((content: string) => {
    setMessages((prev) => {
      const next = [...prev]
      const last = next[next.length - 1]
      next[next.length - 1] = { ...last, content, error: true }
      return next
    })
  }, [])

  /**
   * Chips are keyed by tool name, one running slot each: a `start` for a name
   * already present resets that slot rather than piling on a second chip, so
   * a tool called twice in one turn just re-runs its own chip in place. An
   * `ok`/`failed` updates the matching slot's status only.
   */
  const updateToolActivity = useCallback((activity: ToolActivity) => {
    setMessages((prev) => {
      const next = [...prev]
      const last = next[next.length - 1]
      const tools = last.tools ? [...last.tools] : []
      const index = tools.findIndex((t) => t.name === activity.name)

      if (index === -1) {
        tools.push(activity)
      } else if (activity.status === 'start') {
        tools[index] = activity
      } else {
        tools[index] = { ...tools[index], status: activity.status }
      }

      next[next.length - 1] = { ...last, tools }
      return next
    })
  }, [])

  const send = useCallback(
    async (text: string) => {
      if (!text || abortRef.current) return

      const controller = new AbortController()
      abortRef.current = controller

      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: 'user', content: text },
        { id: crypto.randomUUID(), role: 'assistant', content: '' },
      ])
      setIsStreaming(true)

      // Assembled alongside the state updates so the completed turn can be
      // persisted without reading it back out of a stale closure.
      let assembled = ''

      try {
        for await (const event of streamChat({
          model,
          userInput: text,
          conversationId: conversationIdRef.current,
          signal: controller.signal,
        })) {
          if (event.type === 'error') {
            const detail = event.message || GENERIC_FAILURE
            failLastMessage(detail)
            void persist({
              prompt_content: text,
              response_content: detail,
              is_success: false,
            })
            return
          }
          if (event.type === 'tool') {
            updateToolActivity(event.activity)
            continue
          }
          assembled += event.text
          setMessages((prev) => {
            const next = [...prev]
            const last = next[next.length - 1]
            next[next.length - 1] = { ...last, content: last.content + event.text }
            return next
          })
        }

        void persist({
          prompt_content: text,
          response_content: assembled,
          is_success: true,
        })
      } catch (error) {
        // An abort is the user leaving, not a failure - nothing to record.
        if (isAbort(error)) return
        failLastMessage(GENERIC_FAILURE)
        void persist({
          prompt_content: text,
          response_content: GENERIC_FAILURE,
          is_success: false,
          status_code: error instanceof ApiError ? error.status : undefined,
        })
      } finally {
        abortRef.current = null
        setIsStreaming(false)
      }
    },
    [failLastMessage, model, persist, updateToolActivity],
  )

  // Never leave a stream running against an unmounted component.
  useEffect(() => () => abortRef.current?.abort(), [])

  return { messages, isStreaming, send }
}
