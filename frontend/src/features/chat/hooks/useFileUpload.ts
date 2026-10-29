import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError, isAbort, messageFrom } from '../../../lib/http'
import { deleteDocument, getAttachedDocument, uploadPdf } from '../api'
import type { UploadedFile } from '../types'

// Mirrors `_ACCEPTED` in backend/app/application/ingestion/use_cases/upload_document.py,
// by hand. The server still decides. This just saves a round trip.
const ACCEPTED_TYPES = new Set([
  'application/pdf',
  'text/plain',
  'text/markdown',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
])

/**
 * Owns the document attachment for one conversation.
 *
 * Uploads name the thread. Opening a thread reads what the server holds. Late
 * responses are checked against their thread, so a switch mid-upload does not
 * show the old chip.
 */
export function useFileUpload(conversationId: number) {
  const [file, setFile] = useState<UploadedFile | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  // Upload limit deadline, or null. The budget is small, so users hit it.
  const [cooldownUntil, setCooldownUntil] = useState<number | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  // The current thread, to check late responses. Set in an effect for StrictMode.
  const conversationRef = useRef(conversationId)
  useEffect(() => {
    conversationRef.current = conversationId
  }, [conversationId])

  // Wraps every upload: abort, errors, cooldown, and the chip on success.
  const run = useCallback(
    async (task: (signal: AbortSignal) => Promise<UploadedFile>, forConversation: number) => {
      abortRef.current?.abort()
      const controller = new AbortController()
      abortRef.current = controller

      setError(null)
      setIsUploading(true)

      try {
        const uploaded = await task(controller.signal)
        // Only show the chip if the user is still on that thread. The upload
        // keeps running and lands in its own thread either way.
        if (conversationRef.current === forConversation) {
          setFile(uploaded)
        }
      } catch (err) {
        if (isAbort(err)) return
        // Same guard: ignore a failure from a thread the user left.
        if (conversationRef.current !== forConversation) return
        // Keep the deadline too, so the countdown stays accurate.
        if (err instanceof ApiError && err.status === 429) {
          setCooldownUntil(Date.now() + (err.retryAfterSeconds ?? 60) * 1000)
        }
        setError(messageFrom(err, 'Failed to attach that.'))
      } finally {
        if (abortRef.current === controller) {
          abortRef.current = null
          setIsUploading(false)
        }
      }
    },
    [],
  )

  const upload = useCallback(
    async (selected: File) => {
      if (!ACCEPTED_TYPES.has(selected.type)) {
        setError('File must be a PDF, text, markdown, or Word (.docx) document.')
        return
      }
      await run((signal) => uploadPdf(selected, conversationId, signal), conversationId)
    },
    [run, conversationId],
  )

  /**
   * Deletes the attachment: passages, stored file and row.
   *
   * The chip clears only after the server confirms. On failure it stays, so
   * pressing again retries.
   */
  const clear = useCallback(async () => {
    const attached = file
    setError(null)
    if (attached === null) return

    try {
      await deleteDocument(attached.id)
      setFile(null)
    } catch (err) {
      setError(messageFrom(err, 'Could not remove that file.'))
    }
  }, [file])

  // Load the attachment when a thread opens. A 404 (a stale pin) means none.
  // Other failures are reported, since "unknown" is not "none".
  useEffect(() => {
    let cancelled = false
    setFile(null)
    setError(null)
    getAttachedDocument(conversationId)
      .then((attached) => {
        if (!cancelled) setFile(attached)
      })
      .catch((err) => {
        if (cancelled) return
        if (err instanceof ApiError && err.status === 404) {
          setFile(null)
          return
        }
        setError(messageFrom(err, 'Could not check for an attached file.'))
      })
    return () => {
      cancelled = true
    }
  }, [conversationId])

  useEffect(() => () => abortRef.current?.abort(), [])

  return { file, isUploading, error, cooldownUntil, upload, clear }
}