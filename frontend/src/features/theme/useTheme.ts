import { useCallback, useEffect, useState } from 'react'

export type Theme = 'ivory' | 'mono'

const THEME_KEY = 'theme'

function getInitialTheme(): Theme {
  const stored = localStorage.getItem(THEME_KEY)
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

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem(THEME_KEY, theme)
  }, [theme])

  const toggle = useCallback(
    () => setTheme((current) => (current === 'mono' ? 'ivory' : 'mono')),
    [],
  )

  return { theme, toggle }
}
