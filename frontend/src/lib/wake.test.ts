import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ATTEMPT_TIMEOUT_MS, WAKE_DEADLINE_MS, waitForBackend } from './wake'

/** A `/health` answer from the real backend. */
function healthy(): Response {
  return new Response(JSON.stringify({ status: 'ok' }), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  })
}

/** What a misconfigured proxy returns: the SPA shell, with a 200. */
function spaFallback(): Response {
  return new Response('<!doctype html><title>Talk to web</title>', {
    status: 200,
    headers: { 'Content-Type': 'text/html' },
  })
}

const fetchMock = vi.fn<typeof fetch>()

beforeEach(() => {
  vi.useFakeTimers()
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe('waitForBackend', () => {
  it('returns awake on the first answer, without retrying', async () => {
    fetchMock.mockResolvedValue(healthy())

    const result = waitForBackend()
    await vi.runAllTimersAsync()

    await expect(result).resolves.toBe('awake')
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('keeps probing while the backend is still starting', async () => {
    fetchMock
      .mockRejectedValueOnce(new TypeError('Failed to fetch'))
      .mockResolvedValueOnce(new Response('', { status: 502 }))
      .mockResolvedValueOnce(healthy())

    const result = waitForBackend()
    await vi.runAllTimersAsync()

    await expect(result).resolves.toBe('awake')
    expect(fetchMock).toHaveBeenCalledTimes(3)
  })

  it('does not accept the SPA fallback as a live backend', async () => {
    fetchMock.mockResolvedValueOnce(spaFallback()).mockResolvedValueOnce(healthy())

    const result = waitForBackend()
    await vi.runAllTimersAsync()

    await expect(result).resolves.toBe('awake')
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('gives up once the deadline passes', async () => {
    fetchMock.mockRejectedValue(new TypeError('Failed to fetch'))

    const result = waitForBackend()
    await vi.advanceTimersByTimeAsync(WAKE_DEADLINE_MS + ATTEMPT_TIMEOUT_MS)

    await expect(result).resolves.toBe('unreachable')
  })

  it('stops when the caller aborts', async () => {
    const controller = new AbortController()
    fetchMock.mockRejectedValue(new TypeError('Failed to fetch'))

    const result = waitForBackend({ signal: controller.signal })
    await vi.advanceTimersByTimeAsync(0)
    const callsBeforeAbort = fetchMock.mock.calls.length

    controller.abort()
    await vi.runAllTimersAsync()

    await expect(result).resolves.toBe('unreachable')
    expect(fetchMock).toHaveBeenCalledTimes(callsBeforeAbort)
  })

  it('reports every attempt so the caller can show progress', async () => {
    const onAttempt = vi.fn()
    fetchMock.mockRejectedValueOnce(new TypeError('Failed to fetch')).mockResolvedValue(healthy())

    const result = waitForBackend({ onAttempt })
    await vi.runAllTimersAsync()

    await expect(result).resolves.toBe('awake')
    expect(onAttempt).toHaveBeenCalledTimes(2)
  })
})
