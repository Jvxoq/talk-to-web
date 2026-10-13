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

/** How often a document that is still indexing is read again. */
const POLL_INTERVAL_MS = 2_000

/** How long a failed upload stays on screen before its chip closes itself. */
export const FAILED_VISIBLE_MS = 4_000

/**
 * What the attachment chip shows.
 *
 * `uploading` is the request in flight. `indexing` is a stored file the
 * server is still making searchable - the upload answers before that runs, so
 * a 200 alone is not "complete". Both read as "Uploading" to the user.
 *
 * A `failed` attachment with an `id` failed while indexing: the server holds a
 * row for it, which is deleted so the chip does not come back on the next
 * open. One with no `id` was refused before it was stored.
 */
export type Attachment =
  | { status: 'uploading'; name: string }
  | { status: 'indexing' | 'ready'; name: string; id: number }
  | { status: 'failed'; name: string; id: number | null; reason: string | null }

function fromServer(document: UploadedFile | null): Attachment | null {
  if (document === null) return null
  const { id, name, status } = document
  if (status === 'failed') return { status, name, id, reason: null }
  return { status, name, id }
}

// A 4xx carries this app's own sentence about what was wrong with the file,
// which is worth showing. A 5xx or a network failure says nothing the user can
// act on, so the chip's "Upload failed" stands alone.
function reasonFor(error: unknown): string | null {
  if (error instanceof ApiError && error.status >= 400 && error.status < 500) {
    return error.message
  }
  return null
}

/**
 * Owns the document attachment for one conversation.
 *
 * The chip mirrors the server. A document belongs to the thread it was
 * attached to and outlives the tab that uploaded it, so opening a thread reads
 * what the server still holds. A new upload is shown as uploading until the
 * server says it is searchable - polled, because indexing runs after the
 * upload has answered - and as failed, briefly, if it never becomes so.
 *
 * Every late response is checked against the thread it was for, because
 * switching threads mid-upload must not paste the old thread's chip onto the
 * new one.
 */
export function useFileUpload(conversationId: number) {
  const [attachment, setAttachment] = useState<Attachment | null>(null)
  const [error, setError] = useState<string | null>(null)
  // Absolute deadline for the upload limit, or null. Uploads are the expensive
  // half of this app — every accepted file is embedded — so the budget is small
  // and users meet it.
  const [cooldownUntil, setCooldownUntil] = useState<number | null>(null)
  // Bumped to read the thread's attachment again without changing thread.
  const [reads, setReads] = useState(0)
  const abortRef = useRef<AbortController | null>(null)
  // The thread each in-flight step was started for, so a late completion can
  // be compared against the thread this hook now shows. Updated in an effect,
  // not at render: a ref written during render goes stale under StrictMode's
  // double render.
  const conversationRef = useRef(conversationId)
  useEffect(() => {
    conversationRef.current = conversationId
  }, [conversationId])

  const upload = useCallback(
    async (selected: File) => {
      const forConversation = conversationId
      if (!ACCEPTED_TYPES.has(selected.type)) {
        setAttachment({
          status: 'failed',
          name: selected.name,
          id: null,
          reason: 'File must be a PDF, text, markdown, or Word (.docx) document.',
        })
        return
      }

      abortRef.current?.abort()
      const controller = new AbortController()
      abortRef.current = controller

      setError(null)
      setAttachment({ status: 'uploading', name: selected.name })

      try {
        const uploaded = await uploadPdf(selected, forConversation, controller.signal)
        // The upload named `forConversation`. If the user has moved to another
        // thread since, its chip is no longer this hook's business: the
        // document still belongs to the thread it was uploaded into, and that
        // thread's own read will show it when the user returns.
        if (conversationRef.current === forConversation) {
          setAttachment(fromServer(uploaded))
        }
      } catch (err) {
        if (isAbort(err)) return
        // Same guard as the success path: a failure that lands after the user
        // has moved on belongs to the thread they left.
        if (conversationRef.current !== forConversation) return
        // The deadline is kept so the countdown can be rendered from it.
        if (err instanceof ApiError && err.status === 429) {
          setCooldownUntil(Date.now() + (err.retryAfterSeconds ?? 60) * 1000)
        }
        setAttachment({ status: 'failed', name: selected.name, id: null, reason: reasonFor(err) })
      } finally {
        if (abortRef.current === controller) abortRef.current = null
      }
    },
    [conversationId],
  )

  /**
   * Removes the attachment for good: passages, stored file and row.
   *
   * The chip only goes when the server agrees, because it mirrors what the
   * server holds. A failed delete leaves the chip up, which is the honest
   * rendering of "still attached", and makes pressing again the obvious retry.
   */
  const clear = useCallback(async () => {
    const attached = attachment
    setError(null)
    if (attached?.status !== 'ready') return

    try {
      await deleteDocument(attached.id)
      setAttachment((current) => (current === attached ? null : current))
    } catch (err) {
      setError(messageFrom(err, 'Could not remove that file.'))
    }
  }, [attachment])

  // Opening a thread loads its attachment. A 404 is folded into "none": the
  // server answers 404 for a thread id that is not this account's, which a
  // stale pinned id can name for a moment before the bootstrap replaces it -
  // noise, not something to report. Other failures leave the chip unset and
  // say so, because "nothing attached" and "unknown" are different states and
  // only one of them is safe to show.
  useEffect(() => {
    let cancelled = false
    setAttachment(null)
    setError(null)
    getAttachedDocument(conversationId)
      .then((attached) => {
        if (cancelled) return
        // An upload started while this read was in flight is newer than
        // anything the read can say.
        setAttachment((current) => (current?.status === 'uploading' ? current : fromServer(attached)))
      })
      .catch((err) => {
        if (cancelled) return
        if (err instanceof ApiError && err.status === 404) return
        setError(messageFrom(err, 'Could not check for an attached file.'))
      })
    return () => {
      cancelled = true
    }
  }, [conversationId, reads])

  // While the server is indexing, read the attachment again until it settles.
  // Each read sets a new object, which is what schedules the next one. A read
  // that fails is retried on the next tick: the server's own deadline turns a
  // run that died into "failed", so this does not wait forever on the server.
  useEffect(() => {
    if (attachment?.status !== 'indexing') return
    const watched = attachment
    let cancelled = false
    const timer = setTimeout(() => {
      getAttachedDocument(conversationId).then(
        (latest) => {
          if (cancelled) return
          setAttachment((current) => (current === watched ? fromServer(latest) : current))
        },
        (err: unknown) => {
          if (cancelled) return
          const gone = err instanceof ApiError && err.status === 404
          setAttachment((current) => (current === watched ? (gone ? null : { ...watched }) : current))
        },
      )
    }, POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [attachment, conversationId])

  // A failed upload is shown, then closes itself.
  useEffect(() => {
    if (attachment?.status !== 'failed') return
    const failed = attachment
    // Indexing failed after the row was written. Removing the row keeps the
    // chip from coming back on the next open. Best effort: if it fails, that
    // next open reads it as failed again and tries again.
    if (failed.id !== null) {
      deleteDocument(failed.id).catch(() => undefined)
    }
    const timer = setTimeout(() => {
      setAttachment((current) => (current === failed ? null : current))
      // Refused before it was stored, so whatever the thread had attached
      // before is still there. Read it back rather than guess.
      if (failed.id === null) setReads((count) => count + 1)
    }, FAILED_VISIBLE_MS)
    return () => clearTimeout(timer)
  }, [attachment])

  useEffect(() => () => abortRef.current?.abort(), [])

  const isUploading = attachment?.status === 'uploading' || attachment?.status === 'indexing'

  return { attachment, isUploading, error, cooldownUntil, upload, clear }
}
