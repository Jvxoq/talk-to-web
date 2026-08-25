import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError, isAbort, messageFrom } from '../../../lib/http'
import { deleteDocument, uploadPdf } from '../api'
import type { UploadedFile } from '../types'

// Mirrors the backend's `_ACCEPTED` table in
// backend/app/application/ingestion/use_cases/upload_document.py — kept in
// sync by hand, the same way the frontend's MIN_PASSWORD_LENGTH tracks the
// domain value object. The server is still the source of truth: this only
// saves a round trip on an obviously wrong file.
const ACCEPTED_TYPES = new Set([
  'application/pdf',
  'text/plain',
  'text/markdown',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
])

/**
 * Owns the document attachment for one conversation.
 *
 * A document belongs to the thread it was attached to, so this hook takes the
 * conversation id and does two things with it. Every upload names it, and
 * switching threads drops the chip - the file it named is not this thread's
 * attachment, and showing it would claim the model can read something it
 * cannot.
 */
export function useFileUpload(conversationId: number) {
  const [file, setFile] = useState<UploadedFile | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  // Absolute deadline for the upload limit, or null. Uploads are the expensive
  // half of this app — every accepted file is embedded — so the budget is small
  // and users meet it.
  const [cooldownUntil, setCooldownUntil] = useState<number | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  // Wraps every upload: abort handling, error and cooldown surfacing, and the
  // attachment chip on success.
  const run = useCallback(async (task: (signal: AbortSignal) => Promise<UploadedFile>) => {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setError(null)
    setIsUploading(true)

    try {
      setFile(await task(controller.signal))
    } catch (err) {
      if (isAbort(err)) return
      // The message is already the server's own sentence; what it cannot do is
      // stay true as the seconds pass, so the deadline is kept alongside it and
      // the countdown is rendered from that.
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
  }, [])

  const upload = useCallback(
    async (selected: File) => {
      if (!ACCEPTED_TYPES.has(selected.type)) {
        setError('File must be a PDF, text, markdown, or Word (.docx) document.')
        return
      }
      await run((signal) => uploadPdf(selected, conversationId, signal))
    },
    [run, conversationId],
  )

  /**
   * Removes the attachment for good: passages, stored file and row.
   *
   * The chip disappears first and does not come back if the request fails. The
   * user asked for this document to be gone, and leaving the chip up to report
   * a server-side failure would invite them to keep pressing a button that has
   * already done what it can. The failure is surfaced in its place instead.
   */
  const clear = useCallback(async () => {
    const attached = file
    setFile(null)
    setError(null)
    if (attached === null) return

    try {
      await deleteDocument(attached.id)
    } catch (err) {
      setError(messageFrom(err, 'Could not remove that file.'))
    }
  }, [file])

  // Switching threads is not a removal: the document stays attached to the
  // conversation it was uploaded into, and this only stops showing it here.
  useEffect(() => {
    setFile(null)
    setError(null)
  }, [conversationId])

  useEffect(() => () => abortRef.current?.abort(), [])

  return { file, isUploading, error, cooldownUntil, upload, clear }
}
