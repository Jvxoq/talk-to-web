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

  it('an upload lands on the chip for the thread it was uploaded into', async () => {
    mockedUpload.mockResolvedValue({ id: 5, name: 'new.pdf' })
    const { result } = render(1)

    await act(async () => {
      await result.current.upload(new File(['x'], 'new.pdf', { type: 'application/pdf' }))
    })

    expect(mockedUpload).toHaveBeenCalledWith(expect.any(File), 1, expect.anything())
    expect(result.current.file).toEqual({ id: 5, name: 'new.pdf' })
  })
})
