// @vitest-environment jsdom
// The hook owns React state, so it runs in jsdom via this per-file pragma -
// the rest of the suite stays in the node environment, which is what its
// plain-module tests want.

import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ApiError } from '../../../lib/http'
import { deleteDocument, getAttachedDocument, uploadPdf } from '../api'
import { useFileUpload } from './useFileUpload'

vi.mock('../api', () => ({
  uploadPdf: vi.fn(),
  deleteDocument: vi.fn(),
  getAttachedDocument: vi.fn(),
}))

const mockedUpload = vi.mocked(uploadPdf)
const mockedDelete = vi.mocked(deleteDocument)
const mockedGet = vi.mocked(getAttachedDocument)

function render(conversationId: number) {
  return renderHook((id: number) => useFileUpload(id), { initialProps: conversationId })
}

beforeEach(() => {
  vi.clearAllMocks()
  mockedGet.mockResolvedValue(null)
})

afterEach(() => {
  vi.useRealTimers()
})

describe('useFileUpload', () => {
  it('shows the document the thread still has attached', async () => {
    mockedGet.mockResolvedValue({ id: 3, name: 'quarterly.pdf' })

    const { result } = render(1)

    await waitFor(() => expect(result.current.file).toEqual({ id: 3, name: 'quarterly.pdf' }))
    expect(mockedGet).toHaveBeenCalledWith(1)
  })

  it('shows nothing for a thread with no attachment', async () => {
    const { result } = render(1)

    await waitFor(() => expect(mockedGet).toHaveBeenCalled())
    expect(result.current.file).toBeNull()
    expect(result.current.error).toBeNull()
  })

  it('reads the attachment again when the conversation changes', async () => {
    mockedGet.mockResolvedValueOnce({ id: 3, name: 'a.pdf' })
    const { result, rerender } = render(1)
    await waitFor(() => expect(result.current.file).not.toBeNull())

    mockedGet.mockResolvedValueOnce({ id: 9, name: 'b.pdf' })
    await act(async () => {
      rerender(2)
    })

    await waitFor(() => expect(result.current.file).toEqual({ id: 9, name: 'b.pdf' }))
  })

  it('reads a stale thread id as no attachment, not as an error', async () => {
    // The server answers 404 for a thread id that is not this account's, and
    // a pinned id can name one of those for a moment before the bootstrap
    // replaces it. Noise, not something to show.
    mockedGet.mockRejectedValue(new ApiError(404, 'Unknown conversation'))

    const { result } = render(1)

    await waitFor(() => expect(mockedGet).toHaveBeenCalled())
    expect(result.current.file).toBeNull()
    expect(result.current.error).toBeNull()
  })

  it('reports a failed attachment read instead of quietly showing none', async () => {
    mockedGet.mockRejectedValue(new ApiError(500, 'backend down'))

    const { result } = render(1)

    await waitFor(() => expect(result.current.error).toBe('backend down'))
    expect(result.current.file).toBeNull()
  })

  it('clear removes the chip when the server agrees', async () => {
    mockedGet.mockResolvedValue({ id: 3, name: 'a.pdf' })
    mockedDelete.mockResolvedValue(undefined)
    const { result } = render(1)
    await waitFor(() => expect(result.current.file).not.toBeNull())

    await act(async () => {
      await result.current.clear()
    })

    expect(mockedDelete).toHaveBeenCalledWith(3)
    expect(result.current.file).toBeNull()
  })

  it('clear keeps the chip when the delete fails, since it still exists', async () => {
    // The chip mirrors what the server holds - it is filled from the server on
    // thread open - so clearing it before the server has would show a state
    // that is not true. Keeping it makes pressing again the obvious retry.
    mockedGet.mockResolvedValue({ id: 3, name: 'a.pdf' })
    mockedDelete.mockRejectedValue(new ApiError(500, 'storage refused'))
    const { result } = render(1)
    await waitFor(() => expect(result.current.file).not.toBeNull())

    await act(async () => {
      await result.current.clear()
    })

    expect(result.current.file).toEqual({ id: 3, name: 'a.pdf' })
    expect(result.current.error).toBe('storage refused')
  })

  it('an upload lands on the chip for the thread it was uploaded into', async () => {
    mockedUpload.mockResolvedValue({ id: 5, name: 'new.pdf' })
    const { result } = render(1)

    await act(async () => {
      await result.current.upload(new File(['x'], 'new.pdf', { type: 'application/pdf' }))
    })

    expect(mockedUpload).toHaveBeenCalledWith(expect.any(File), 1, expect.anything())
    expect(result.current.file).toEqual({ id: 5, name: 'new.pdf' })
  })

  it('a late upload response never lands on the chip of the thread switched to', async () => {
    let resolveUpload: (value: { id: number; name: string }) => void = () => {}
    mockedUpload.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveUpload = resolve
        }),
    )
    mockedGet.mockResolvedValue(null)
    const { result, rerender } = render(1)

    let uploadPromise: Promise<void> = Promise.resolve()
    act(() => {
      uploadPromise = result.current.upload(new File(['x'], 'new.pdf', { type: 'application/pdf' }))
    })

    // Switch away before the upload finishes. The upload named thread 1; its
    // chip is no longer this hook's business.
    await act(async () => {
      rerender(2)
    })
    // One read per thread open: the initial render's and the switch's.
    await waitFor(() => expect(mockedGet).toHaveBeenCalledTimes(2))
    resolveUpload({ id: 5, name: 'new.pdf' })
    await act(async () => {
      await uploadPromise
    })

    expect(result.current.file).toBeNull()
    expect(result.current.isUploading).toBe(false)
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
    mockedGet.mockResolvedValue(null)
    const { result, rerender } = render(1)

    let uploadPromise: Promise<void> = Promise.resolve()
    act(() => {
      uploadPromise = result.current.upload(new File(['x'], 'new.pdf', { type: 'application/pdf' }))
    })
    await act(async () => {
      rerender(2)
    })
    await waitFor(() => expect(mockedGet).toHaveBeenCalledTimes(2))
    rejectUpload(new ApiError(500, 'backend down'))
    await act(async () => {
      await uploadPromise
    })

    expect(result.current.error).toBeNull()
  })

  it('an upload that fails on the current thread is reported', async () => {
    mockedUpload.mockRejectedValue(new ApiError(500, 'backend down'))
    const { result } = render(1)

    await act(async () => {
      await result.current.upload(new File(['x'], 'new.pdf', { type: 'application/pdf' }))
    })

    expect(result.current.error).toBe('backend down')
    expect(result.current.isUploading).toBe(false)
  })
})
