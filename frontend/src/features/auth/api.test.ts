import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { signOut } from './api'
import { getOrCreateConversationId, storeConversationId } from '../../lib/conversation'

/**
 * A minimal `localStorage`, since node has none and the tests would pass vacuously.
 */
function fakeStorage(): Storage {
  const entries = new Map<string, string>()
  return {
    get length() {
      return entries.size
    },
    key: (index: number) => [...entries.keys()][index] ?? null,
    getItem: (key: string) => entries.get(key) ?? null,
    setItem: (key: string, value: string) => void entries.set(key, value),
    removeItem: (key: string) => void entries.delete(key),
    clear: () => entries.clear(),
  }
}

beforeEach(() => {
  vi.stubGlobal('localStorage', fakeStorage())
  vi.stubGlobal(
    'fetch',
    vi.fn<typeof fetch>().mockResolvedValue(new Response(null, { status: 204 })),
  )
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('signOut', () => {
  it('unpins the conversation, so the next account does not inherit it', async () => {
    storeConversationId(42)

    await signOut()

    expect(getOrCreateConversationId()).toBeNull()
  })

  it('unpins it even when the logout request fails', async () => {
    storeConversationId(42)
    vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockRejectedValue(new TypeError('offline')))

    await expect(signOut()).rejects.toThrow()

    expect(getOrCreateConversationId()).toBeNull()
  })
})
