import { describe, expect, it } from 'vitest'
import { formatDuration, formatTokens, parseUsage } from './usage'

describe('parseUsage', () => {
  it('narrows a well-formed usage frame payload', () => {
    const raw = {
      prompt_tokens: 812,
      completion_tokens: 143,
      elapsed_ms: 1834,
      model: 'qwen/qwen3.6-27b',
    }

    expect(parseUsage(raw)).toEqual({
      promptTokens: 812,
      completionTokens: 143,
      elapsedMs: 1834,
      model: 'qwen/qwen3.6-27b',
    })
  })

  it('rejects a non-object payload', () => {
    expect(parseUsage(null)).toBeUndefined()
    expect(parseUsage('usage')).toBeUndefined()
    expect(parseUsage(42)).toBeUndefined()
  })

  it('rejects a payload missing or mistyping a required field', () => {
    const base = {
      prompt_tokens: 10,
      completion_tokens: 5,
      elapsed_ms: 250,
      model: 'some/model',
    }

    expect(parseUsage({ ...base, prompt_tokens: '10' })).toBeUndefined()
    expect(parseUsage({ ...base, elapsed_ms: '250' })).toBeUndefined()
    expect(parseUsage({ ...base, model: undefined })).toBeUndefined()
    const { elapsed_ms: _elapsedMs, ...missingElapsed } = base
    expect(parseUsage(missingElapsed)).toBeUndefined()
  })
})

describe('formatTokens', () => {
  it('sums prompt and completion tokens into a plain count', () => {
    expect(
      formatTokens({
        promptTokens: 812,
        completionTokens: 143,
        elapsedMs: 1834,
        model: 'm',
      }),
    ).toBe('955 tokens')
  })
})

describe('formatDuration', () => {
  it('shows whole milliseconds below one second', () => {
    expect(
      formatDuration({
        promptTokens: 1,
        completionTokens: 1,
        elapsedMs: 420,
        model: 'm',
      }),
    ).toBe('420ms')
  })

  it('shows one decimal place of seconds at or above one second', () => {
    expect(
      formatDuration({
        promptTokens: 1,
        completionTokens: 1,
        elapsedMs: 1834,
        model: 'm',
      }),
    ).toBe('1.8s')
  })
})
