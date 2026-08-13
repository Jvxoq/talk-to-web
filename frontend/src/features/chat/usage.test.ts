import { describe, expect, it } from 'vitest'
import { formatCost, formatTokens, parseUsage } from './usage'

describe('parseUsage', () => {
  it('narrows a well-formed usage frame payload', () => {
    const raw = {
      prompt_tokens: 812,
      completion_tokens: 143,
      cost_usd: 0.000208,
      model: 'openai/gpt-oss-120b',
      priced: true,
    }

    expect(parseUsage(raw)).toEqual({
      promptTokens: 812,
      completionTokens: 143,
      costUsd: 0.000208,
      model: 'openai/gpt-oss-120b',
      priced: true,
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
      cost_usd: 0.0001,
      model: 'some/model',
      priced: true,
    }

    expect(parseUsage({ ...base, prompt_tokens: '10' })).toBeUndefined()
    expect(parseUsage({ ...base, cost_usd: '0.0001' })).toBeUndefined()
    expect(parseUsage({ ...base, model: undefined })).toBeUndefined()
    expect(parseUsage({ ...base, priced: 1 })).toBeUndefined()
    const { priced: _priced, ...missingPriced } = base
    expect(parseUsage(missingPriced)).toBeUndefined()
  })
})

describe('formatTokens', () => {
  it('sums prompt and completion tokens into a plain count', () => {
    expect(
      formatTokens({
        promptTokens: 812,
        completionTokens: 143,
        costUsd: 0.000208,
        model: 'm',
        priced: true,
      }),
    ).toBe('955 tokens')
  })
})

describe('formatCost', () => {
  it('shows six decimal places for a sub-cent priced reply', () => {
    expect(
      formatCost({
        promptTokens: 1,
        completionTokens: 1,
        costUsd: 0.000208,
        model: 'm',
        priced: true,
      }),
    ).toBe('$0.000208')
  })

  it('shows two decimal places once the cost clears a cent', () => {
    expect(
      formatCost({
        promptTokens: 1,
        completionTokens: 1,
        costUsd: 0.42,
        model: 'm',
        priced: true,
      }),
    ).toBe('$0.42')
  })

  it('never renders $0.00 for an unpriced reply — priced: false means no price was on file, not that it was free', () => {
    expect(
      formatCost({
        promptTokens: 1,
        completionTokens: 1,
        costUsd: 0,
        model: 'm',
        priced: false,
      }),
    ).toBe('cost unknown')
  })
})
