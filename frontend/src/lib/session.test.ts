import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ApiError } from './http'

type SessionModule = typeof import('./session')

/**
 * The access token and the in-flight refresh are module-level state, which is
 * the point of the design — so every test gets a fresh copy of the module
 * rather than trying to reset that state from the outside.
 */
let session: SessionModule
let fetchMock: ReturnType<typeof vi.fn>

/** A `/auth/refresh` body in the shape the server actually sends. */
function sessionBody(token: string): string {
  return JSON.stringify({
    access_token: token,
    token_type: 'bearer',
    user: { id: 7, email: 'a@example.com' },
  })
}

function jsonResponse(body: string, status = 200): Response {
  return new Response(body, { status, headers: { 'content-type': 'application/json' } })
}

/** A promise that only settles when the test says so. */
function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void; reject: (e: unknown) => void } {
  let resolve!: (value: T) => void
  let reject!: (e: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

beforeEach(async () => {
  vi.resetModules()
  fetchMock = vi.fn()
  vi.stubGlobal('fetch', fetchMock)
  session = await import('./session')
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('parseSession', () => {
  it('narrows a well-formed body', () => {
    expect(session.parseSession(JSON.parse(sessionBody('abc')))).toEqual({
      accessToken: 'abc',
      user: { id: 7, email: 'a@example.com' },
    })
  })

  it.each([
    ['not an object', null],
    ['a string', 'abc'],
    ['no access_token', { user: { id: 1, email: 'a@b.c' } }],
    ['a non-string access_token', { access_token: 1, user: { id: 1, email: 'a@b.c' } }],
    ['no user', { access_token: 'abc' }],
    ['a non-numeric user id', { access_token: 'abc', user: { id: '1', email: 'a@b.c' } }],
    ['no email', { access_token: 'abc', user: { id: 1 } }],
  ])('rejects a body with %s', (_label, raw) => {
    expect(() => session.parseSession(raw)).toThrow()
  })
})

describe('authEndpoint', () => {
  it('joins the default base to a path', () => {
    expect(session.authEndpoint('refresh')).toBe('/auth/refresh')
  })
})

describe('session token', () => {
  it('starts empty and follows set/clear', () => {
    expect(session.getAccessToken()).toBeNull()
    session.setAccessToken('t1')
    expect(session.getAccessToken()).toBe('t1')
    session.clearAccessToken()
    expect(session.getAccessToken()).toBeNull()
  })

  it('notifies subscribers until they unsubscribe', () => {
    const listener = vi.fn()
    const unsubscribe = session.subscribeToSession(listener)

    session.setAccessToken('t1')
    expect(listener).toHaveBeenCalledTimes(1)

    unsubscribe()
    session.setAccessToken('t2')
    expect(listener).toHaveBeenCalledTimes(1)
  })
})

describe('refreshSession', () => {
  it('sends the refresh cookie and stores the new token', async () => {
    fetchMock.mockResolvedValue(jsonResponse(sessionBody('fresh')))

    const result = await session.refreshSession()

    expect(result.accessToken).toBe('fresh')
    expect(session.getAccessToken()).toBe('fresh')
    expect(fetchMock).toHaveBeenCalledWith('/auth/refresh', {
      method: 'POST',
      credentials: 'include',
    })
  })

  it('makes one request for concurrent callers', async () => {
    const gate = deferred<Response>()
    fetchMock.mockReturnValue(gate.promise)

    const calls = [session.refreshSession(), session.refreshSession(), session.refreshSession()]
    gate.resolve(jsonResponse(sessionBody('fresh')))

    const results = await Promise.all(calls)
    expect(fetchMock).toHaveBeenCalledTimes(1)
    // Rotation spends the presented refresh token, so a second request would
    // look like reuse to the server and revoke every session the user has.
    expect(results.map((r) => r.accessToken)).toEqual(['fresh', 'fresh', 'fresh'])
  })

  it('starts a new request once the previous one has settled', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(sessionBody('one')))
    fetchMock.mockResolvedValueOnce(jsonResponse(sessionBody('two')))

    expect((await session.refreshSession()).accessToken).toBe('one')
    expect((await session.refreshSession()).accessToken).toBe('two')
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('clears the token and rejects when the cookie is no longer good', async () => {
    session.setAccessToken('stale')
    fetchMock.mockResolvedValue(jsonResponse('{}', 401))

    await expect(session.refreshSession()).rejects.toMatchObject({ status: 401 })
    expect(session.getAccessToken()).toBeNull()
  })

  it('rejects every concurrent caller of a failed refresh', async () => {
    const gate = deferred<Response>()
    fetchMock.mockReturnValue(gate.promise)

    const calls = [
      session.refreshSession().catch((e: ApiError) => e.status),
      session.refreshSession().catch((e: ApiError) => e.status),
    ]
    gate.resolve(jsonResponse('{}', 401))

    expect(await Promise.all(calls)).toEqual([401, 401])
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('recovers after a failure instead of caching it', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse('{}', 401))
    await expect(session.refreshSession()).rejects.toThrow()

    fetchMock.mockResolvedValueOnce(jsonResponse(sessionBody('fresh')))
    expect((await session.refreshSession()).accessToken).toBe('fresh')
  })
})

describe('authorizedFetch', () => {
  it('attaches the bearer token and keeps the caller headers', async () => {
    session.setAccessToken('t1')
    fetchMock.mockResolvedValue(jsonResponse('{}'))

    await session.authorizedFetch('/conversations', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
    })

    expect(fetchMock).toHaveBeenCalledWith('/conversations', {
      method: 'POST',
      headers: { 'content-type': 'application/json', Authorization: 'Bearer t1' },
    })
  })

  it('sends no Authorization header when there is no token', async () => {
    fetchMock.mockResolvedValue(jsonResponse('{}'))

    await session.authorizedFetch('/conversations')

    expect(fetchMock).toHaveBeenCalledWith('/conversations', { headers: {} })
  })

  it('does not refresh on a non-401 failure', async () => {
    session.setAccessToken('t1')
    fetchMock.mockResolvedValue(jsonResponse('{}', 500))

    const response = await session.authorizedFetch('/conversations')

    expect(response.status).toBe(500)
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('refreshes once and replays the request on a 401', async () => {
    session.setAccessToken('stale')
    fetchMock
      .mockResolvedValueOnce(jsonResponse('{}', 401))
      .mockResolvedValueOnce(jsonResponse(sessionBody('fresh')))
      .mockResolvedValueOnce(jsonResponse('{"ok":true}'))

    const response = await session.authorizedFetch('/conversations')

    expect(response.status).toBe(200)
    expect(fetchMock).toHaveBeenCalledTimes(3)
    expect(fetchMock.mock.calls[2]).toEqual([
      '/conversations',
      { headers: { Authorization: 'Bearer fresh' } },
    ])
  })

  it('does not retry a second time when the replay is also a 401', async () => {
    session.setAccessToken('stale')
    fetchMock
      .mockResolvedValueOnce(jsonResponse('{}', 401))
      .mockResolvedValueOnce(jsonResponse(sessionBody('fresh')))
      .mockResolvedValueOnce(jsonResponse('{}', 401))

    const response = await session.authorizedFetch('/conversations')

    expect(response.status).toBe(401)
    expect(fetchMock).toHaveBeenCalledTimes(3)
  })

  it('signs out and returns the original 401 when the refresh fails', async () => {
    session.setAccessToken('stale')
    fetchMock
      .mockResolvedValueOnce(jsonResponse('{}', 401))
      .mockResolvedValueOnce(jsonResponse('{}', 401))

    const response = await session.authorizedFetch('/conversations')

    expect(response.status).toBe(401)
    expect(session.getAccessToken()).toBeNull()
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('refreshes once for a burst of concurrent 401s', async () => {
    session.setAccessToken('stale')
    fetchMock.mockImplementation((url: string) =>
      Promise.resolve(
        url === '/auth/refresh'
          ? jsonResponse(sessionBody('fresh'))
          : jsonResponse('{}', 401),
      ),
    )

    // Three requests fail at once. Each wants a token; only one rotation may
    // happen, or the server sees the spent token replayed and revokes the lot.
    await Promise.all([
      session.authorizedFetch('/a'),
      session.authorizedFetch('/b'),
      session.authorizedFetch('/c'),
    ])

    const refreshes = fetchMock.mock.calls.filter(([url]) => url === '/auth/refresh')
    expect(refreshes).toHaveLength(1)
  })
})
