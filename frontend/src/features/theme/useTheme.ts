import { useCallback, useLayoutEffect, useRef, useState } from 'react'
import { flushSync } from 'react-dom'

export type Theme = 'ivory' | 'mono'

const THEME_KEY = 'theme'

/**
 * Storage access can throw (blocked cookies, some private modes). This hook
 * runs above the error boundary, so a throw here would blank the page: treat
 * unreadable storage as "nothing saved" and an unwritable one as "not saved".
 */
function readStoredTheme(): string | null {
  try {
    return localStorage.getItem(THEME_KEY)
  } catch {
    return null
  }
}

function storeTheme(theme: Theme): void {
  try {
    localStorage.setItem(THEME_KEY, theme)
  } catch {
    // The choice just won't survive a reload.
  }
}

function getInitialTheme(): Theme {
  const stored = readStoredTheme()
  // Old saved names from the previous design map onto the new themes: the
  // black-and-white dark mode is Mono, the warm paper is Ivory.
  if (stored === 'mono' || stored === 'dark') return 'mono'
  if (stored === 'ivory' || stored === 'light') return 'ivory'
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'mono' : 'ivory'
}

/**
 * Mono is the black-and-white dark theme of the same design system; Ivory is
 * the warm paper one. The swap is a whole token block in globals.css, so
 * components never branch on theme — they only ever read tokens.
 */
export function useTheme() {
  const [theme, setTheme] = useState<Theme>(getInitialTheme)
  // The newest crossfade. A second click starts a new one and skips the first;
  // only the newest may turn per-element transitions back on.
  const latest = useRef<ViewTransition | null>(null)

  // A layout effect, so the attribute lands in the same commit `flushSync`
  // forces inside the view transition, before the new snapshot is taken.
  useLayoutEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    storeTheme(theme)
  }, [theme])

  const toggle = useCallback(() => {
    // An updater, not the closed-over theme: a second click can land before
    // the first transition's callback has committed, and it must flip the
    // theme the first click produced, not the one this render saw.
    const flip = (current: Theme): Theme => (current === 'mono' ? 'ivory' : 'mono')
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (reduce || !document.startViewTransition) {
      setTheme(flip)
      return
    }
    // One snapshot crossfade for the whole page, so the background, inputs,
    // buttons and borders all change together instead of each on its own
    // transition timing. `theme-switching` pauses those per-element
    // transitions so the live page underneath lands on the new colours at once.
    const root = document.documentElement
    root.classList.add('theme-switching')
    const transition = document.startViewTransition(() => {
      flushSync(() => setTheme(flip))
    })
    latest.current = transition
    // Both settle paths: `finished` rejects if the swap itself threw, and that
    // must neither leave transitions paused nor surface as an unhandled rejection.
    const done = () => {
      if (latest.current === transition) root.classList.remove('theme-switching')
    }
    void transition.finished.then(done, done)
  }, [])

  return { theme, toggle }
}
