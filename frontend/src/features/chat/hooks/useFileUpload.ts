import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError, isAbort, messageFrom } from '../../../lib/http'
import { deleteDocument, getAttachedDocument, uploadPdf } from '../api'
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
 * A document belongs to the thread it was attached to and outlives the tab
 * that uploaded it, so this hook does three things with the conversation id.
 * Every upload names it. Opening a thread reads what the server still holds,
 * so the chip shows an attachment made in an earlier session too - a chip
 * that only remembered this session's uploads would hide a file the model
 * can still be asked about, and leave nothing to press to remove it. And
 * every late response is checked against the thread it was for, because
 * switching threads mid-upload must not paste the old thread's chip onto the
 * new one.
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
  // The thread each in-flight step was started for, so a late completion can
  // be compared against the thread this hook now shows. Updated in an effect,
  // not at render: a ref written during render goes stale under StrictMode's
  // double render.
  const conversationRef = useRef(conversationId)
  useEffect(() => {
    conversationRef.current = conversationId
  }, [conversationId])

  // Wraps every upload: abort handling, error and cooldown surfacing, and the
  // attachment chip on success.
  const run = useCallback(
    async (task: (signal: AbortSignal) => Promise<UploadedFile>, forConversation: number) => {
      abortRef.current?.abort()
      const controller = new AbortController()
      abortRef.current = controller

      setError(null)
      setIsUploading(true)

      try {
        const uploaded = await task(controller.signal)
        // The upload named `forConversation`. If the user has moved to another
        // thread since, its chip is no longer this hook's business: the
        // document still belongs to the thread it was uploaded into, and that
        // thread's own read below will show it when the user returns. The
        // request itself is left running - aborting it could leave a partial
        // file behind, and the document would still arrive in the old thread.
        if (conversationRef.current === forConversation) {
          setFile(uploaded)
        }
      } catch (err) {
        if (isAbort(err)) return
        // Same guard as the success path: a failure that lands after the user
        // has moved on belongs to the thread they left, not the one they are
        // on. When they go back, the read below will show what the server
        // actually kept - which, for a failed upload, is nothing.
        if (conversationRef.current !== forConversation) return
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

  // Opening a thread loads its attachment. A 404 is folded into "none": the
  // server answers 404 for a thread id that is not this account's, which a
  // stale pinned id can name for a moment before the bootstrap replaces it -
  // noise, not something to report. Other failures leave the chip unset and
  // say so, because "nothing attached" and "unknown" are different states and
  // only one of them is safe to show.
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
