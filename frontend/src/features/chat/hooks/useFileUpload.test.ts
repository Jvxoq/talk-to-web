// @vitest-environment jsdom
// The hook owns React state, so it runs in jsdom via this per-file pragma -
// the rest of the suite stays in the node environment, which is what its
// plain-module tests want.

import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ApiError } from '../../../lib/http'
import { deleteDocument, getAttachedDocument, uploadPdf } from '../api'
import type { UploadedFile } from '../types'
import { FAILED_VISIBLE_MS, useFileUpload } from './useFileUpload'

vi.mock('../api', () => ({
  uploadPdf: vi.fn(),
  deleteDocument: vi.fn(),
  getAttachedDocument: vi.fn(),
}))

const mockedUpload = vi.mocked(uploadPdf)
const mockedDelete = vi.mocked(deleteDocument)
const mockedGet = vi.mocked(getAttachedDocument)

// Longer than one poll interval, shorter than two, so each call is one read.
const ONE_POLL_MS = 2_500

function render(conversationId: number) {
  return renderHook((id: number) => useFileUpload(id), { initialProps: conversationId })
}

/** Lets pending promises settle and runs timers due within `ms`. */
async function advance(ms = 0) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms)
  })
}

function doc(id: number, name: string, status: UploadedFile['status']): UploadedFile {
  return { id, name, status }
}

function pdf(name = 'new.pdf') {
  return new File(['x'], name, { type: 'application/pdf' })
}

/** A thread with nothing attached, and an upload the server stored and is indexing. */
async function uploadedAndIndexing() {
  mockedUpload.mockResolvedValue(doc(5, 'new.pdf', 'indexing'))
  const hook = render(1)
  await advance()
  await act(async () => {
    await hook.result.current.upload(pdf())
  })
  return hook
}

beforeEach(() => {
  vi.useFakeTimers()
  // Reset, not clear: a `mockResolvedValueOnce` a test left unused would
  // otherwise answer the next test's first read.
  vi.resetAllMocks()
  mockedGet.mockResolvedValue(null)
  mockedDelete.mockResolvedValue(undefined)
})

afterEach(() => {
  vi.useRealTimers()
})

describe('useFileUpload', () => {
  it('shows the document the thread still has attached', async () => {
    mockedGet.mockResolvedValue(doc(3, 'quarterly.pdf', 'ready'))

    const { result } = render(1)
    await advance()

    expect(result.current.attachment).toEqual({ status: 'ready', id: 3, name: 'quarterly.pdf' })
    expect(mockedGet).toHaveBeenCalledWith(1)
  })

  it('shows nothing for a thread with no attachment', async () => {
    const { result } = render(1)
    await advance()

    expect(result.current.attachment).toBeNull()
    expect(result.current.error).toBeNull()
  })

  it('reads the attachment again when the conversation changes', async () => {
    mockedGet.mockResolvedValueOnce(doc(3, 'a.pdf', 'ready'))
    const { result, rerender } = render(1)
    await advance()

    mockedGet.mockResolvedValueOnce(doc(9, 'b.pdf', 'ready'))
    rerender(2)
    await advance()

    expect(result.current.attachment).toEqual({ status: 'ready', id: 9, name: 'b.pdf' })
  })

  it('reads a stale thread id as no attachment, not as an error', async () => {
    // The server answers 404 for a thread id that is not this account's, and
    // a pinned id can name one of those for a moment before the bootstrap
    // replaces it. Noise, not something to show.
    mockedGet.mockRejectedValue(new ApiError(404, 'Unknown conversation'))

    const { result } = render(1)
    await advance()

    expect(result.current.attachment).toBeNull()
    expect(result.current.error).toBeNull()
  })

  it('reports a failed attachment read instead of quietly showing none', async () => {
    mockedGet.mockRejectedValue(new ApiError(500, 'backend down'))

    const { result } = render(1)
    await advance()

    expect(result.current.error).toBe('backend down')
    expect(result.current.attachment).toBeNull()
  })

  it('clear removes the chip when the server agrees', async () => {
    mockedGet.mockResolvedValue(doc(3, 'a.pdf', 'ready'))
    const { result } = render(1)
    await advance()

    await act(async () => {
      await result.current.clear()
    })

    expect(mockedDelete).toHaveBeenCalledWith(3)
    expect(result.current.attachment).toBeNull()
  })

  it('clear keeps the chip when the delete fails, since it still exists', async () => {
    // The chip mirrors what the server holds, so clearing it before the server
    // has would show a state that is not true.
    mockedGet.mockResolvedValue(doc(3, 'a.pdf', 'ready'))
    mockedDelete.mockRejectedValue(new ApiError(500, 'storage refused'))
    const { result } = render(1)
    await advance()

    await act(async () => {
      await result.current.clear()
    })

    expect(result.current.attachment).toEqual({ status: 'ready', id: 3, name: 'a.pdf' })
    expect(result.current.error).toBe('storage refused')
  })

  it('shows uploading while the request is in flight', async () => {
    mockedUpload.mockReturnValue(new Promise(() => {}))
    const { result } = render(1)
    await advance()

    act(() => {
      void result.current.upload(pdf())
    })

    expect(result.current.attachment).toEqual({ status: 'uploading', name: 'new.pdf' })
    expect(result.current.isUploading).toBe(true)
    expect(mockedUpload).toHaveBeenCalledWith(expect.any(File), 1, expect.any(AbortSignal))
  })

  it('stays uploading until the server finishes indexing', async () => {
    // The upload answers before indexing runs. Showing the file as complete
    // on that answer is what put a chip up for a file the model could not read.
    const { result } = await uploadedAndIndexing()

    expect(result.current.attachment).toEqual({ status: 'indexing', id: 5, name: 'new.pdf' })
    expect(result.current.isUploading).toBe(true)
  })

  it('shows upload complete once indexing succeeds', async () => {
    const { result } = await uploadedAndIndexing()

    mockedGet.mockResolvedValueOnce(doc(5, 'new.pdf', 'indexing'))
    await advance(ONE_POLL_MS)
    mockedGet.mockResolvedValueOnce(doc(5, 'new.pdf', 'ready'))
    await advance(ONE_POLL_MS)

    expect(result.current.attachment).toEqual({ status: 'ready', id: 5, name: 'new.pdf' })
    expect(result.current.isUploading).toBe(false)
  })

  it('stops polling once the document settles', async () => {
    await uploadedAndIndexing()
    mockedGet.mockResolvedValue(doc(5, 'new.pdf', 'ready'))
    await advance(ONE_POLL_MS)
    const readsWhenSettled = mockedGet.mock.calls.length

    await advance(ONE_POLL_MS * 5)

    expect(mockedGet).toHaveBeenCalledTimes(readsWhenSettled)
  })

  it('keeps polling through a failed read', async () => {
    const { result } = await uploadedAndIndexing()

    mockedGet.mockRejectedValueOnce(new ApiError(503, 'waking up'))
    await advance(ONE_POLL_MS)
    mockedGet.mockResolvedValueOnce(doc(5, 'new.pdf', 'ready'))
    await advance(ONE_POLL_MS)

    expect(result.current.attachment).toEqual({ status: 'ready', id: 5, name: 'new.pdf' })
  })

  it('resumes polling for a thread opened mid-indexing', async () => {
    mockedGet.mockResolvedValueOnce(doc(5, 'new.pdf', 'indexing'))
    const { result } = render(1)
    await advance()

    mockedGet.mockResolvedValueOnce(doc(5, 'new.pdf', 'ready'))
    await advance(ONE_POLL_MS)

    expect(result.current.attachment).toEqual({ status: 'ready', id: 5, name: 'new.pdf' })
  })

  it('shows upload failed when indexing fails', async () => {
    const { result } = await uploadedAndIndexing()

    mockedGet.mockResolvedValueOnce(doc(5, 'new.pdf', 'failed'))
    await advance(ONE_POLL_MS)

    expect(result.current.attachment).toEqual({
      status: 'failed',
      id: 5,
      name: 'new.pdf',
      reason: null,
    })
  })

  it('removes the row of a document that failed indexing', async () => {
    // Left in place, the failed row would bring the chip back on every open.
    mockedGet.mockResolvedValueOnce(doc(5, 'new.pdf', 'failed'))

    render(1)
    await advance()

    expect(mockedDelete).toHaveBeenCalledWith(5)
  })

  it('closes a failed chip by itself', async () => {
    mockedGet.mockResolvedValueOnce(doc(5, 'new.pdf', 'failed'))
    const { result } = render(1)
    await advance()

    await advance(FAILED_VISIBLE_MS)

    expect(result.current.attachment).toBeNull()
  })

  it('keeps a failed chip up until its time is over', async () => {
    mockedGet.mockResolvedValueOnce(doc(5, 'new.pdf', 'failed'))
    const { result } = render(1)
    await advance()

    await advance(FAILED_VISIBLE_MS - 100)

    expect(result.current.attachment?.status).toBe('failed')
  })

  it("shows the server's reason when it refuses the file", async () => {
    mockedUpload.mockRejectedValue(new ApiError(415, 'That file is not a real PDF.'))
    const { result } = render(1)
    await advance()

    await act(async () => {
      await result.current.upload(pdf())
    })

    expect(result.current.attachment).toEqual({
      status: 'failed',
      id: null,
      name: 'new.pdf',
      reason: 'That file is not a real PDF.',
    })
    expect(result.current.isUploading).toBe(false)
  })

  it('hides the text of a server error', async () => {
    mockedUpload.mockRejectedValue(new ApiError(500, 'Internal server error'))
    const { result } = render(1)
    await advance()

    await act(async () => {
      await result.current.upload(pdf())
    })

    expect(result.current.attachment).toEqual({
      status: 'failed',
      id: null,
      name: 'new.pdf',
      reason: null,
    })
  })

  it('brings back the previous file after a refused upload closes', async () => {
    // A refused upload never reached storage, so the file the thread had
    // before is still attached. The chip reads it back from the server.
    mockedGet.mockResolvedValue(doc(3, 'old.pdf', 'ready'))
    mockedUpload.mockRejectedValue(new ApiError(500, 'Internal server error'))
    const { result } = render(1)
    await advance()
    await act(async () => {
      await result.current.upload(pdf())
    })

    await advance(FAILED_VISIBLE_MS)

    expect(result.current.attachment).toEqual({ status: 'ready', id: 3, name: 'old.pdf' })
  })

  it('refuses an unsupported file type without sending it', async () => {
    const { result } = render(1)
    await advance()

    await act(async () => {
      await result.current.upload(new File(['x'], 'tool.exe', { type: 'application/x-msdownload' }))
    })

    expect(mockedUpload).not.toHaveBeenCalled()
    expect(result.current.attachment).toEqual({
      status: 'failed',
      id: null,
      name: 'tool.exe',
      reason: 'File must be a PDF, text, markdown, or Word (.docx) document.',
    })
  })

  it('counts down the upload limit after a 429', async () => {
    vi.setSystemTime(new Date('2026-10-13T00:00:00Z'))
    mockedUpload.mockRejectedValue(new ApiError(429, 'Too many uploads.', 30))
    const { result } = render(1)
    await advance()

    await act(async () => {
      await result.current.upload(pdf())
    })

    expect(result.current.cooldownUntil).toBe(Date.parse('2026-10-13T00:00:30Z'))
  })

  it('a late upload response never lands on the chip of the thread switched to', async () => {
    let resolveUpload: (value: UploadedFile) => void = () => {}
    mockedUpload.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveUpload = resolve
        }),
    )
    const { result, rerender } = render(1)
    await advance()

    let uploadPromise: Promise<void> = Promise.resolve()
    act(() => {
      uploadPromise = result.current.upload(pdf())
    })
    // Switch away before the upload finishes. The upload named thread 1; its
    // chip is no longer this hook's business.
    rerender(2)
    await advance()
    resolveUpload(doc(5, 'new.pdf', 'indexing'))
    await act(async () => {
      await uploadPromise
    })

    expect(result.current.attachment).toBeNull()
  })

  it('an upload that fails after a thread switch is not reported on the new thread', async () => {
    // Held open until after the switch: a rejection that settled first would
    // land on thread 1 and say nothing about the guard.
    let rejectUpload: (reason: unknown) => void = () => {}
    mockedUpload.mockImplementation(
      () =>
        new Promise((_, reject) => {
          rejectUpload = reject
        }),
    )
    const { result, rerender } = render(1)
    await advance()

    let uploadPromise: Promise<void> = Promise.resolve()
    act(() => {
      uploadPromise = result.current.upload(pdf())
    })
    rerender(2)
    await advance()
    rejectUpload(new ApiError(500, 'backend down'))
    await act(async () => {
      await uploadPromise
    })

    expect(result.current.attachment).toBeNull()
  })

  it('an attachment read that lands mid-upload does not replace the upload', async () => {
    let resolveRead: (value: UploadedFile | null) => void = () => {}
    mockedGet.mockReturnValueOnce(
      new Promise((resolve) => {
        resolveRead = resolve
      }),
    )
    mockedUpload.mockReturnValue(new Promise(() => {}))
    const { result } = render(1)

    act(() => {
      void result.current.upload(pdf())
    })
    resolveRead(doc(3, 'old.pdf', 'ready'))
    await advance()

    expect(result.current.attachment).toEqual({ status: 'uploading', name: 'new.pdf' })
  })
})
