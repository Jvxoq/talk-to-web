import { beforeEach, describe, expect, it, vi } from 'vitest'

import { ApiError } from '../../lib/http'
import { getAttachedDocument } from './api'

// The response boundary: how wire shapes are narrowed, and what is refused.
vi.mock('../../lib/session', () => ({
  authorizedFetch: vi.fn(),
}))

import { authorizedFetch } from '../../lib/session'

const mockedFetch = vi.mocked(authorizedFetch)

function ok(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200 })
}

beforeEach(() => {
  vi.clearAllMocks()
})

describe('getAttachedDocument', () => {
  it('reads the document the server names, narrowed to what the chip needs', async () => {
    mockedFetch.mockResolvedValue(
      ok({ id: 3, name: 'a.pdf', chunks_indexed: 12, created_at: '2026-08-06T00:00:00Z' }),
    )

    const attached = await getAttachedDocument(1)

    expect(mockedFetch).toHaveBeenCalledWith('/conversations/1/document')
    expect(attached).toEqual({ id: 3, name: 'a.pdf' })
  })

  it('reads a null body as no attachment', async () => {
    mockedFetch.mockResolvedValue(ok(null))

    expect(await getAttachedDocument(1)).toBeNull()
  })

  it('refuses a body without the fields the chip needs', async () => {
    mockedFetch.mockResolvedValue(ok({ name: 'a.pdf' }))

    await expect(getAttachedDocument(1)).rejects.toThrow('missing "id"')
  })

  it('raises the server status for a failure, so callers can read it', async () => {
    mockedFetch.mockResolvedValue(new Response('{"detail":"nope"}', { status: 404 }))

    const error = await getAttachedDocument(1).then(
      () => null,
      (err: unknown) => err,
    )

    expect(error).toBeInstanceOf(ApiError)
    expect((error as ApiError).status).toBe(404)
  })
})