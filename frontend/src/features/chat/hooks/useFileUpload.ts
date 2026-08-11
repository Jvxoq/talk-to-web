import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError, isAbort, messageFrom } from '../../../lib/http'
import { ingestUrl, uploadPdf } from '../api'
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

/** Owns the document attachment: file selection or a pasted URL, and the error surfaced inline. */
export function useFileUpload() {
  const [file, setFile] = useState<UploadedFile | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  // Absolute deadline for the upload limit, or null. Uploads are the expensive
  // half of this app — every accepted file is embedded — so the budget is small
  // and users meet it.
  const [cooldownUntil, setCooldownUntil] = useState<number | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  // Shared by both `upload` and `ingestUrl`: same abort handling, same error
  // and cooldown surfacing, same attachment chip on success — a URL and a file
  // are two ways to reach the same result.
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
      await run((signal) => uploadPdf(selected, signal))
    },
    [run],
  )

  const ingestUrlAttachment = useCallback(
    async (url: string) => {
      await run((signal) => ingestUrl(url, signal))
    },
    [run],
  )

  const clear = useCallback(() => {
    setFile(null)
    setError(null)
  }, [])

  useEffect(() => () => abortRef.current?.abort(), [])

  return { file, isUploading, error, cooldownUntil, upload, ingestUrl: ingestUrlAttachment, clear }
}
