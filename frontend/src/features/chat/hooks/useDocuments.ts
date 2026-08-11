import { useCallback, useState } from 'react'
import { deleteDocument, fetchDocuments } from '../api'
import { messageFrom } from '../../../lib/http'
import type { DocumentSummary } from '../types'

/**
 * Owns the document manager panel's list, loaded on demand.
 *
 * Fetched lazily rather than alongside the conversation list on sign-in: most
 * sessions never open the panel, and every one of them would otherwise pay
 * for a request whose answer is thrown away unread.
 */
export function useDocuments() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [deletingId, setDeletingId] = useState<number | null>(null)

  const refresh = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    try {
      setDocuments(await fetchDocuments())
    } catch (err) {
      setError(messageFrom(err, 'Could not load your documents.'))
    } finally {
      setIsLoading(false)
    }
  }, [])

  const remove = useCallback(async (id: number) => {
    setDeletingId(id)
    setError(null)
    try {
      await deleteDocument(id)
      setDocuments((prev) => prev.filter((d) => d.id !== id))
    } catch (err) {
      setError(messageFrom(err, 'Could not delete that document.'))
    } finally {
      setDeletingId(null)
    }
  }, [])

  return { documents, isLoading, error, deletingId, refresh, remove }
}
