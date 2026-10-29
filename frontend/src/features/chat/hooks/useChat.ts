import { useCallback, useEffect, useRef, useState } from 'react'
import {
  appendMessage,
  getConversation,
  type ConversationOut,
  type MessageCreate,
  type MessageOut,
} from '../../../lib/conversation'
import { ApiError, isAbort } from '../../../lib/http'
import { streamChat } from '../api'
import type { Message, Model, ToolActivity } from '../types'

const GENERIC_FAILURE = 'Something went wrong. Please try again.'

/**
 * Turn each stored prompt/response row into two bubbles, with stable ids.
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
  * Owns the transcript and the in-flight stream for one conversation.
  *
  * Changing `conversationId` reloads the transcript and aborts the old stream.
  * The assistant turn is appended empty and filled as deltas arrive. Finished
  * turns are saved to the server, best-effort. `preloaded` is used once, on
  * first load.
  */
export function useChat(model: Model, conversationId: number, preloaded?: ConversationOut | null) {
  const [messages, setMessages] = useState<Message[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  // An absolute deadline (see `useCountdown`), or null. Per account, so a
  // conversation switch does not reset it.
  const [cooldownUntil, setCooldownUntil] = useState<number | null>(null)
  // The refused message, returned to the composer so it is not lost.
  const [restoreText, setRestoreText] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const cooldownRef = useRef<number | null>(null)
  // A ref, so a new object from the parent does not re-run the load.
  const preloadedRef = useRef(preloaded)
  preloadedRef.current = preloaded
  // Which conversation ids have already had their preload spent.
  const consumedRef = useRef<Set<number>>(new Set())
  // The id last loaded, so StrictMode's rerun does not fetch twice.
  const loadStartedForRef = useRef<number | null>(null)

  useEffect(() => {
    // Checked before any reset, or StrictMode's remount would clear `messages`.
    if (loadStartedForRef.current === conversationId) return
    loadStartedForRef.current = conversationId

    // Abort a reply still streaming into the thread being left.
    abortRef.current?.abort()
    abortRef.current = null
    setIsStreaming(false)
    setMessages([])

    const ready = preloadedRef.current
    if (ready && ready.id === conversationId && !consumedRef.current.has(conversationId)) {
      consumedRef.current.add(conversationId)
      setMessages(toMessages(ready.messages))
      return
    }

    let cancelled = false
    getConversation(conversationId)
      .then((conversation) => {
        if (!cancelled) setMessages(toMessages(conversation.messages))
      })
      .catch(() => {
        // Swallowed: an empty composer is still usable.
      })

    return () => {
      cancelled = true
    }
  }, [conversationId])

  const persist = useCallback(
    async (message: MessageCreate) => {
      try {
        await appendMessage(conversationId, message)
      } catch {
        // Swallowed: a failed save should not interrupt the chat.
      }
    },
    [conversationId],
  )

  const failLastMessage = useCallback((content: string) => {
    setMessages((prev) => {
      const next = [...prev]
      const last = next[next.length - 1]
      next[next.length - 1] = { ...last, content, error: true }
      return next
    })
  }, [])

  /**
   * One chip per tool name. A repeat `start` resets that chip. `ok` or
   * `failed` updates its status and sources.
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
        tools[index] = { ...tools[index], status: activity.status, sources: activity.sources }
      }

      next[next.length - 1] = { ...last, tools }
      return next
    })
  }, [])

  const send = useCallback(
    async (text: string) => {
      if (!text || abortRef.current) return

      // Also checked here, not just on the button, to avoid a second refusal.
      if (cooldownRef.current !== null && cooldownRef.current > Date.now()) {
        setRestoreText(text)
        return
      }

      const controller = new AbortController()
      abortRef.current = controller

      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: 'user', content: text },
        { id: crypto.randomUUID(), role: 'assistant', content: '' },
      ])
      setIsStreaming(true)

      // Built alongside state, to save the turn without a stale closure.
      let assembled = ''

      try {
        for await (const event of streamChat({
          model,
          userInput: text,
          conversationId,
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
          if (event.type === 'summarizing') {
            // One slot per turn, overwritten, since a reply can condense twice.
            setMessages((prev) => {
              const next = [...prev]
              const last = next[next.length - 1]
              next[next.length - 1] = { ...last, summarizing: event.summarizing }
              return next
            })
            continue
          }
          if (event.type === 'usage') {
            setMessages((prev) => {
              const next = [...prev]
              const last = next[next.length - 1]
              next[next.length - 1] = { ...last, usage: event.usage }
              return next
            })
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

        // A rate limit gets its own message, since "try again" is the wrong advice.
        const limited = error instanceof ApiError && error.status === 429
        const detail = limited ? (error as ApiError).message : GENERIC_FAILURE

        if (limited) {
          const seconds = (error as ApiError).retryAfterSeconds ?? 60
          const until = Date.now() + seconds * 1000
          cooldownRef.current = until
          setCooldownUntil(until)
          setRestoreText(text)
        }

        failLastMessage(detail)
        void persist({
          prompt_content: text,
          response_content: detail,
          is_success: false,
          status_code: error instanceof ApiError ? error.status : undefined,
        })
      } finally {
        abortRef.current = null
        setIsStreaming(false)
      }
    },
    [conversationId, failLastMessage, model, persist, updateToolActivity],
  )

  // Never leave a stream running against an unmounted component.
  useEffect(() => () => abortRef.current?.abort(), [])

  const clearRestoreText = useCallback(() => setRestoreText(null), [])

  return { messages, isStreaming, send, cooldownUntil, restoreText, clearRestoreText }
}
