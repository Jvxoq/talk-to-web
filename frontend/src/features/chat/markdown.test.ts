import { describe, expect, it } from 'vitest'

import { normalizeStreamingMarkdown } from './markdown'

/**
 * Each case is a stream cut mid-block. The check is "this frame does not look broken".
 */
describe('normalizeStreamingMarkdown', () => {
  it('leaves settled text alone', () => {
    const text = '# Title\n\nA paragraph.\n\n- one\n- two'
    expect(normalizeStreamingMarkdown(text)).toBe(text)
  })

  it('is a no-op on the empty string', () => {
    expect(normalizeStreamingMarkdown('')).toBe('')
  })

  describe('code fences', () => {
    it('closes an unclosed backtick fence', () => {
      const text = 'Here:\n```py\nprint(1)'
      expect(normalizeStreamingMarkdown(text)).toBe('Here:\n```py\nprint(1)\n```')
    })

    it('closes an unclosed tilde fence', () => {
      expect(normalizeStreamingMarkdown('~~~\nx')).toBe('~~~\nx\n```')
    })

    it('leaves a balanced fence alone', () => {
      const text = '```js\nconst a = 1\n```\n\nDone.'
      expect(normalizeStreamingMarkdown(text)).toBe(text)
    })

    it('closes the second fence when a third opens', () => {
      const text = '```\na\n```\n```\nb'
      expect(normalizeStreamingMarkdown(text)).toBe(`${text}\n\`\`\``)
    })

    it('counts an indented fence', () => {
      expect(normalizeStreamingMarkdown('  ```\n  x')).toBe('  ```\n  x\n```')
    })
  })

  describe('trailing horizontal rules', () => {
    it('drops a trailing rule that would underline the paragraph above', () => {
      expect(normalizeStreamingMarkdown('Heading?\n---')).toBe('Heading?')
    })

    it.each(['===', '***', '___'])('drops a trailing %s', (rule) => {
      expect(normalizeStreamingMarkdown(`Text\n${rule}`)).toBe('Text')
    })

    it('keeps a rule once a line has arrived after it', () => {
      const text = 'Text\n---\nMore'
      expect(normalizeStreamingMarkdown(text)).toBe(text)
    })

    it('keeps a rule followed by a blank line', () => {
      const text = 'Text\n---\n'
      expect(normalizeStreamingMarkdown(text)).toBe(text)
    })
  })

  describe('tables', () => {
    it('hides a header row with no delimiter yet', () => {
      expect(normalizeStreamingMarkdown('Intro\n| a | b |')).toBe('Intro')
    })

    it('hides a header whose delimiter row is still being written', () => {
      expect(normalizeStreamingMarkdown('Intro\n| a | b |\n|---|--')).toBe('Intro')
    })

    it('shows the table once the delimiter row is settled by a following line', () => {
      const text = 'Intro\n| a | b |\n| --- | --- |\n| 1 | 2 |'
      expect(normalizeStreamingMarkdown(text)).toBe(text)
    })

    it('hides a table whose last line is the delimiter row', () => {
      // A delimiter on the last line may still be incomplete.
      expect(normalizeStreamingMarkdown('Intro\n| a | b |\n| --- | --- |')).toBe('Intro')
    })

    it('drops the whole table when it starts the message', () => {
      expect(normalizeStreamingMarkdown('| a | b |')).toBe('')
    })

    it('leaves text that merely mentions a pipe alone', () => {
      const text = 'Use a | b to pipe.'
      expect(normalizeStreamingMarkdown(text)).toBe(text)
    })
  })

  it('drops a trailing rule and closes a fence in the same pass', () => {
    expect(normalizeStreamingMarkdown('```\ncode\n---')).toBe('```\ncode\n```')
  })
})
