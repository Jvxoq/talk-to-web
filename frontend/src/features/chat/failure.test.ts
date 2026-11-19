import { describe, expect, it } from 'vitest'
import { failureMessage, parseFailure } from './failure'

describe('parseFailure', () => {
  it.each(['timeout', 'overloaded', 'unavailable', 'internal'] as const)(
    'reads the %s code',
    (code) => {
      expect(parseFailure({ code, message: 'server copy' })).toBe(code)
    },
  )

  it('reads an unknown code as internal', () => {
    expect(parseFailure({ code: 'quota_exceeded', message: 'x' })).toBe('internal')
  })

  it('reads an older backend string as internal', () => {
    // An older backend sent the provider's own text here. It must never be shown.
    expect(parseFailure('Error code: 401 - invalid_api_key sk-live-123')).toBe('internal')
  })

  it('ignores inherited keys as codes', () => {
    expect(parseFailure({ code: 'toString' })).toBe('internal')
  })

  it('reads a malformed payload as internal', () => {
    expect(parseFailure(null)).toBe('internal')
    expect(parseFailure({})).toBe('internal')
    expect(parseFailure({ code: 42 })).toBe('internal')
  })
})

describe('failureMessage', () => {
  it('shows our own copy for a code', () => {
    expect(failureMessage('timeout')).toBe(
      'The assistant took too long to answer. Please try again.',
    )
  })
})
