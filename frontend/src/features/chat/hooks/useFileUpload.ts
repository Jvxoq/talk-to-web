import { useCallback, useEffect, useRef, useState } from 'react'
import { isAbort, messageFrom } from '../../../lib/http'
import { uploadPdf } from '../api'
import type { UploadedFile } from '../types'

/** Owns the PDF attachment: selection, upload, and the error surfaced inline. */
export function useFileUpload() {
  const [file, setFile] = useState<UploadedFile | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const upload = useCallback(async (selected: File) => {
    if (selected.type !== 'application/pdf') {
      setError('File must be a PDF.')
      return
    }

    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setError(null)
    setIsUploading(true)

    try {
      setFile(await uploadPdf(selected, controller.signal))
    } catch (err) {
      if (!isAbort(err)) setError(messageFrom(err, 'Failed to upload file.'))
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null
        setIsUploading(false)
      }
    }
  }, [])

  const clear = useCallback(() => {
    setFile(null)
    setError(null)
  }, [])

  useEffect(() => () => abortRef.current?.abort(), [])

  return { file, isUploading, error, upload, clear }
}
