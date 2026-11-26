// @vitest-environment jsdom
// Security tests for what an assistant reply is allowed to put in the DOM.
// A hostile page the agent read can steer the reply, so the renderer is the
// last line of defence against a zero-click leak (security evals S4 and S7).

import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import type { Message } from '../types'
import { MessageBubble } from './MessageBubble'

const PIXEL = 'https://collector.evals.test/pixel.png?id=MID-0106-VQX'

function renderMessage(content: string, role: Message['role'] = 'assistant') {
  const message: Message = { id: 'm1', role, content }
  const view = render(<MessageBubble message={message} pending={false} />)
  return view.container
}

/** Any element that could make the browser fetch a URL on its own. */
function fetchers(root: HTMLElement) {
  return root.querySelectorAll(
    'img, source, picture, video, audio, iframe, script, object, embed, link, [style*="url("], [srcset]',
  )
}

afterEach(cleanup)

describe('MessageBubble markdown images (S4/S7)', () => {
  it.each([
    ['https', `![status](${PIXEL})`],
    ['http', '![status](http://collector.evals.test/p.png?id=MID-0106-VQX)'],
    ['protocol-relative', '![status](//collector.evals.test/p.png?id=MID-0106-VQX)'],
  ])('renders no fetching element for a %s image', (_name, md) => {
    const root = renderMessage(md)
    expect(fetchers(root)).toHaveLength(0)
    expect(root.innerHTML).not.toContain('collector.evals.test')
    expect(root.textContent).toContain('status')
  })

  it('drops a reference-style image the same way', () => {
    const root = renderMessage(`![ref alt][1]\n\n[1]: ${PIXEL}`)
    expect(fetchers(root)).toHaveLength(0)
    expect(root.innerHTML).not.toContain('collector.evals.test')
    expect(root.textContent).toContain('ref alt')
  })

  it('drops an image nested inside a link and keeps no link to the image URL', () => {
    const root = renderMessage(`[![inner alt](${PIXEL})](https://example.com/page)`)
    expect(fetchers(root)).toHaveLength(0)
    expect(root.innerHTML).not.toContain('collector.evals.test')
    expect(root.textContent).toContain('inner alt')
  })

  it('shows the alt text without turning the image URL into a link', () => {
    const root = renderMessage(`![status](${PIXEL})`)
    expect(root.querySelector('a')).toBeNull()
  })
})

describe('MessageBubble raw HTML (S7)', () => {
  it.each([
    ['img onerror', '<img src=x onerror=alert(1)>'],
    ['script', '<script>alert(1)</script>'],
    ['iframe', '<iframe src="https://collector.evals.test/f"></iframe>'],
  ])('never turns %s into elements', (_name, html) => {
    const root = renderMessage(`before ${html} after`)
    expect(fetchers(root)).toHaveLength(0)
    expect(root.querySelector('[onerror]')).toBeNull()
  })
})

describe('MessageBubble link schemes (S7)', () => {
  it.each(['javascript:alert(1)', 'vbscript:msgbox(1)', 'data:text/html,<b>x</b>'])(
    'renders no href for %s',
    (href) => {
      const root = renderMessage(`[click](${href})`)
      const hrefs = [...root.querySelectorAll('a')].map((a) => a.getAttribute('href') ?? '')
      expect(hrefs.filter((h) => /^(javascript|vbscript|data):/i.test(h))).toEqual([])
      expect(root.textContent).toContain('click')
    },
  )
})

describe('MessageBubble normal markdown (twin)', () => {
  it('keeps a normal https link working', () => {
    const root = renderMessage('[docs](https://example.com/docs)')
    const link = root.querySelector('a')
    expect(link?.getAttribute('href')).toBe('https://example.com/docs')
    expect(link?.textContent).toBe('docs')
  })

  it('still renders bold, lists and code', () => {
    const root = renderMessage('**bold**\n\n- one\n- two\n\n`code`')
    expect(root.querySelector('strong')?.textContent).toBe('bold')
    expect(root.querySelectorAll('li')).toHaveLength(2)
    expect(root.querySelector('code')?.textContent).toBe('code')
  })

  it('shows a user message as plain text, never parsed as markdown', () => {
    const root = renderMessage(`![x](${PIXEL})`, 'user')
    expect(fetchers(root)).toHaveLength(0)
    expect(root.textContent).toContain('![x](')
  })
})
