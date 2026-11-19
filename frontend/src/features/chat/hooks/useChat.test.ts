// @vitest-environment jsdom
// The hook owns React state, so it runs in jsdom via this per-file pragma.

import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { getConversation, type ConversationOut } from '../../../lib/conversation'
import { useChat } from './useChat'

vi.mock('../../../lib/conversation', () => ({
  getConversation: vi.fn(),
  appendMessage: vi.fn(),
}))

vi.mock('../api', () => ({ streamChat: vi.fn() }))

const mockedGet = vi.mocked(getConversation)

function conversation(id: number, prompt: string): ConversationOut {
  return {
    id,
    title: `Chat ${id}`,
    model_type: 'test',
    messages: [{ id: id * 10, prompt_content: prompt, response_content: 'ok' }],
  }
}

/** A fetch the test resolves by hand, so it can choose the landing order. */
function deferred() {
  let resolve!: (value: ConversationOut) => void
  const promise = new Promise<ConversationOut>((r) => {
    resolve = r
  })
  return { promise, resolve }
}

afterEach(() => {
  vi.clearAllMocks()
})

describe('useChat history load', () => {
  it('ignores an older load of the same thread that lands after a newer one', async () => {
    const firstA = deferred()
    const b = deferred()
    const secondA = deferred()
    mockedGet
      .mockReturnValueOnce(firstA.promise)
      .mockReturnValueOnce(b.promise)
      .mockReturnValueOnce(secondA.promise)

    const { result, rerender } = renderHook((id: number) => useChat('test', id), {
      initialProps: 1,
    })
    rerender(2)
    rerender(1)

    await act(async () => {
      secondA.resolve(conversation(1, 'newer'))
      await secondA.promise
    })
    expect(result.current.messages[0]?.content).toBe('newer')
    expect(result.current.isLoading).toBe(false)

    await act(async () => {
      firstA.resolve(conversation(1, 'older'))
      b.resolve(conversation(2, 'other'))
      await Promise.all([firstA.promise, b.promise])
    })
    expect(result.current.messages[0]?.content).toBe('newer')
    expect(result.current.isLoading).toBe(false)
  })
})
