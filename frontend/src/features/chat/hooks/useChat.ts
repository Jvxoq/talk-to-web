import { useCallback, useEffect, useRef, useState } from 'react'
import {
  appendMessage,
  getConversation,
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
 * Owns the transcript and the in-flight stream for one, externally-chosen
 * conversation.
 *
 * `conversationId` is now a prop rather than something this hook resolves
 * itself — `useConversations` owns the sidebar's list and which id is active,
 * and this hook's only job is to load and talk into whichever one that is.
 * Switching it is a normal prop change: the effect below reloads the
 * transcript and cuts off any stream still running against the thread the
 * user just left.
 *
 * The assistant's turn is appended empty and filled in place as deltas land, so
 * the list renders a pending bubble without a second piece of state to keep in
 * sync with `messages`.
 *
 * Turns are mirrored to the server as they complete, but that mirror is
 * best-effort: what is on screen stays authoritative for this page-life.
 */
export function useChat(model: Model, conversationId: number) {
  const [messages, setMessages] = useState<Message[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  // An absolute deadline, not a duration: see `useCountdown`. Null when there
  // is no wait, which is the ordinary state. Not reset on a conversation
  // switch - the budget it tracks is per account, not per thread.
  const [cooldownUntil, setCooldownUntil] = useState<number | null>(null)
  // What the user typed when a send was refused, handed back to the composer so
  // a limit they did not know about does not cost them their sentence.
  const [restoreText, setRestoreText] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const cooldownRef = useRef<number | null>(null)

  useEffect(() => {
    // A reply mid-flight belongs to the thread the user is leaving, not the
    // one they are about to see - cut it off rather than let it keep filling
    // a bubble that is no longer on screen.
    abortRef.current?.abort()
    abortRef.current = null
    setIsStreaming(false)
    setMessages([])

    let cancelled = false
    getConversation(conversationId)
      .then((conversation) => {
        if (!cancelled) setMessages(toMessages(conversation.messages))
      })
      .catch(() => {
        // Swallowed: a history that failed to load still leaves a usable,
        // empty composer, which beats blocking the switch entirely.
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
        // Deliberately swallowed: a failed write is invisible to the user, and
        // surfacing it would break up a conversation that otherwise worked.
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
   * Chips are keyed by tool name, one running slot each: a `start` for a name
   * already present resets that slot rather than piling on a second chip, so
   * a tool called twice in one turn just re-runs its own chip in place. An
   * `ok`/`failed` updates the matching slot's status and, when the finished
   * frame carried any, its sources.
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

      // Checked here as well as on the disabled button: the request is certain
      // to be refused while the wait is live, and firing it anyway would show
      // the user a second failure for a limit they are already waiting out.
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

      // Assembled alongside the state updates so the completed turn can be
      // persisted without reading it back out of a stale closure.
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

        // A rate limit is the one failure that is temporary, expected, and has
        // an answer, so it says so in the user's own words rather than through
        // GENERIC_FAILURE. "Something went wrong, please try again" would be
        // both wrong - nothing went wrong - and the worst possible advice.
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
