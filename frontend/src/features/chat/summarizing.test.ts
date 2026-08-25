import { describe, expect, it } from 'vitest'
import { formatTokenCount, parseSummarizing, summarizingText } from './summarizing'

describe('parseSummarizing', () => {
  it('narrows a start frame, which carries no after-count', () => {
    expect(parseSummarizing({ status: 'start', tokens_before: 12480 })).toEqual({
      status: 'start',
      tokensBefore: 12480,
      tokensAfter: undefined,
    })
  })

  it('narrows a done frame with both counts', () => {
    expect(
      parseSummarizing({ status: 'done', tokens_before: 12480, tokens_after: 4100 }),
    ).toEqual({ status: 'done', tokensBefore: 12480, tokensAfter: 4100 })
  })

  it('rejects a payload this build cannot read', () => {
    expect(parseSummarizing(null)).toBeUndefined()
    expect(parseSummarizing('start')).toBeUndefined()
    expect(parseSummarizing({ status: 'paused', tokens_before: 10 })).toBeUndefined()
    expect(parseSummarizing({ status: 'start' })).toBeUndefined()
    expect(parseSummarizing({ status: 'done', tokens_before: 10, tokens_after: '4' })).toBeUndefined()
  })
})

describe('summarizingText', () => {
  it('says what is happening while it runs', () => {
    expect(summarizingText({ status: 'start', tokensBefore: 12480 })).toBe('Condensing context…')
  })

  it('reports both sizes once it is done', () => {
    expect(summarizingText({ status: 'done', tokensBefore: 12480, tokensAfter: 4100 })).toBe(
      'Context condensed 12.5k → 4.1k',
    )
  })

  it('drops the numbers when the after-count never arrived', () => {
    expect(summarizingText({ status: 'done', tokensBefore: 12480 })).toBe('Context condensed')
  })
})

describe('formatTokenCount', () => {
  it('keeps small counts exact and abbreviates the rest', () => {
    expect(formatTokenCount(940)).toBe('940')
    expect(formatTokenCount(1000)).toBe('1.0k')
    expect(formatTokenCount(12480)).toBe('12.5k')
  })
})
