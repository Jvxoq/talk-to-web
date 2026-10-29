import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError } from '../../../lib/http'
import {
  createConversation,
  deleteConversation,
  getConversation,
  getOrCreateConversationId,
  listConversations,
  storeConversationId,
  type ConversationOut,
  type ConversationSummary,
} from '../../../lib/conversation'
import type { Model } from '../types'

/**
 * Owns the sidebar list and the active conversation.
 *
 * The `localStorage` pin is only the starting point on load. An empty list
 * opens a fresh conversation. `enabled` stays false until a user is signed in.
 *
 * The bootstrap fetches the list and the pinned transcript in parallel, and
 * hands the transcript out as `preloaded`. That removes a serial round trip
 * from page load.
 */
export function useConversations(model: Model, enabled: boolean) {
  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const [activeId, setActiveId] = useState<number | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  // The transcript fetched with the list, so `useChat` need not refetch it.
  // Null on a first visit or a stale pin.
  const [preloaded, setPreloaded] = useState<ConversationOut | null>(null)
  // Why the last "new chat" failed, for example the 409 conversation cap.
  const [error, setError] = useState<string | null>(null)

  // A ref, so switching models does not re-run the bootstrap.
  const modelRef = useRef(model)
  useEffect(() => {
    modelRef.current = model
  }, [model])

  // Mirrors `conversations`, so callbacks need not depend on the list.
  const conversationsRef = useRef<ConversationSummary[]>([])
  useEffect(() => {
    conversationsRef.current = conversations
  }, [conversations])

  const bootstrapped = useRef(false)
  // Bumped only on sign-out, so a late bootstrap answer cannot reach the next
  // account. Not bumped per effect run, which would break StrictMode's double mount.
  const generation = useRef(0)

  useEffect(() => {
    if (!enabled) {
      // Signed out: clear the last account's data so the next user never sees it.
      generation.current += 1
      bootstrapped.current = false
      setConversations([])
      setActiveId(null)
      setPreloaded(null)
      setError(null)
      setIsLoading(true)
      return
    }

    // One bootstrap only, even under StrictMode's remount.
    if (bootstrapped.current) return
    bootstrapped.current = true

    const run = generation.current
    /** False once the account this bootstrap belongs to has signed out. */
    const current = () => run === generation.current

    // Synchronous, so it is known before any request.
    const pinned = getOrCreateConversationId()

    Promise.all([
      listConversations(),
      // Speculative. A stale pin wastes one request. Failure resolves to null.
      pinned === null ? Promise.resolve(null) : getConversation(pinned).catch(() => null),
    ])
      .then(async ([items, pinnedConversation]) => {
        if (!current()) return

        if (items.length > 0) {
          const active =
            pinned !== null && items.some((c) => c.id === pinned) ? pinned : items[0].id
          storeConversationId(active)
          setConversations(items)
          // Only if the pin won. Otherwise `useChat` fetches the fallback.
          if (pinnedConversation !== null && pinnedConversation.id === active) {
            setPreloaded(pinnedConversation)
          }
          setActiveId(active)
          return
        }

        const created = await createConversation(modelRef.current)
        // Checked again, since the account may have changed meanwhile.
        if (!current()) return
        storeConversationId(created.id)
        setConversations([created])
        setPreloaded(created)
        setActiveId(created.id)
      })
      .catch(() => {
        // Swallowed: an empty sidebar is still a usable page.
      })
      .finally(() => {
        if (current()) setIsLoading(false)
      })
  }, [enabled])

  const select = useCallback((id: number) => {
    setActiveId(id)
    storeConversationId(id)
  }, [])

  const startNew = useCallback(async () => {
    setError(null)
    let created: ConversationOut
    try {
      created = await createConversation(modelRef.current)
    } catch (err) {
      // 409 is the cap, with the server's own message. Other errors show here too.
      setError(
        err instanceof ApiError ? err.message : 'Could not start a new chat. Try again.',
      )
      return
    }
    // A new conversation is empty, so preload it instead of fetching.
    setPreloaded(created)
    setConversations((prev) => [created, ...prev])
    storeConversationId(created.id)
    setActiveId(created.id)
  }, [])

  const remove = useCallback(
    async (id: number) => {
      await deleteConversation(id)
      // Deleting one makes room, so whatever the cap last said is stale.
      setError(null)
      const remaining = conversationsRef.current.filter((c) => c.id !== id)
      setConversations(remaining)

      if (id !== activeId) return

      if (remaining.length > 0) {
        select(remaining[0].id)
        return
      }

      // The last one is gone, so open a fresh one to type into.
      const created = await createConversation(modelRef.current)
      setConversations([created])
      setPreloaded(created)
      storeConversationId(created.id)
      setActiveId(created.id)
    },
    [activeId, select],
  )

  return { conversations, activeId, isLoading, preloaded, error, select, startNew, remove }
}
