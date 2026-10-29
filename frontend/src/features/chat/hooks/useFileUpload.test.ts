// @vitest-environment jsdom
// This hook test needs jsdom. The rest of the suite runs in node.

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
    // A 404 from a stale pinned id is not an error to show.
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
    // The chip stays until the server confirms, so pressing again retries.
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

    // Switch threads before the upload finishes. Thread 1's chip must not show.
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
    mockedUpload.mockRejectedValue(new ApiError(500, 'backend down'))
    mockedGet.mockResolvedValue(null)
    const { result, rerender } = render(1)

    let uploadPromise: Promise<void> = Promise.resolve()
    act(() => {
      uploadPromise = result.current.upload(new File(['x'], 'new.pdf', { type: 'application/pdf' }))
    })
    await act(async () => {
      rerender(2)
    })
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