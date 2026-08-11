import { useCallback, useEffect, useRef, useState } from 'react'
import {
  createConversation,
  deleteConversation,
  getOrCreateConversationId,
  listConversations,
  storeConversationId,
  type ConversationSummary,
} from '../../../lib/conversation'
import type { Model } from '../types'

/**
 * Owns the sidebar's list and which conversation is active.
 *
 * The pinned id in `localStorage` is now a starting hint rather than the
 * whole story - it picks up where the browser left off on first load, but
 * `select`, `startNew` and `remove` are what change it from here on.
 *
 * An account with no conversations yet still needs one to talk into, so an
 * empty list on first load - or after deleting the last one - opens a fresh
 * conversation the same way this app always has, just from here instead of
 * from `useChat`.
 *
 * `enabled` gates the bootstrap: the caller passes `false` until there is a
 * signed-in user, so this never fires `GET /conversations/` before there is a
 * token to send with it.
 */
export function useConversations(model: Model, enabled: boolean) {
  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const [activeId, setActiveId] = useState<number | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  // Read at bootstrap and inside `startNew` via a ref, so neither re-runs
  // when the user switches models mid-session.
  const modelRef = useRef(model)
  useEffect(() => {
    modelRef.current = model
  }, [model])

  // Mirrors `conversations` so `remove` can compute what is left without
  // making every callback depend on (and be rebuilt by) the list itself.
  const conversationsRef = useRef<ConversationSummary[]>([])
  useEffect(() => {
    conversationsRef.current = conversations
  }, [conversations])

  const bootstrapped = useRef(false)
  useEffect(() => {
    // Waits for `enabled`, then the ref survives StrictMode's remount so
    // exactly one bootstrap runs from that point on.
    if (!enabled || bootstrapped.current) return
    bootstrapped.current = true

    listConversations()
      .then(async (items) => {
        if (items.length > 0) {
          const pinned = getOrCreateConversationId()
          const active = pinned !== null && items.some((c) => c.id === pinned) ? pinned : items[0].id
          storeConversationId(active)
          setConversations(items)
          setActiveId(active)
          return
        }

        const created = await createConversation(modelRef.current)
        storeConversationId(created.id)
        setConversations([created])
        setActiveId(created.id)
      })
      .catch(() => {
        // Swallowed: an empty sidebar is the worst case, and it is still a
        // usable (if conversation-less) shell rather than a broken page.
      })
      .finally(() => setIsLoading(false))
  }, [enabled])

  const select = useCallback((id: number) => {
    setActiveId(id)
    storeConversationId(id)
  }, [])

  const startNew = useCallback(async () => {
    const created = await createConversation(modelRef.current)
    setConversations((prev) => [created, ...prev])
    storeConversationId(created.id)
    setActiveId(created.id)
  }, [])

  const remove = useCallback(
    async (id: number) => {
      await deleteConversation(id)
      const remaining = conversationsRef.current.filter((c) => c.id !== id)
      setConversations(remaining)

      if (id !== activeId) return

      if (remaining.length > 0) {
        select(remaining[0].id)
        return
      }

      // The last conversation is gone - open a fresh one rather than leaving
      // the composer with nothing to send into. `deleteConversation` already
      // cleared the pin; `storeConversationId` below sets the new one.
      const created = await createConversation(modelRef.current)
      setConversations([created])
      storeConversationId(created.id)
      setActiveId(created.id)
    },
    [activeId, select],
  )

  return { conversations, activeId, isLoading, select, startNew, remove }
}
