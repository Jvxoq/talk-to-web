export type ButtonVariant = 'primary' | 'secondary' | 'onDark' | 'ghost'

interface ButtonClassOptions {
  icon?: boolean
  selected?: boolean
  className?: string
}

/**
 * The single source of truth for button styling.
 *
 * Lives apart from the component because interactive buttons can stay plain
 * elements (hover is a colour change in CSS, not a Motion lift). The four
 * variants are fixed by the design system; `icon` and `selected` are modifiers
 * on top of any variant.
 */
export function buttonClass(
  variant: ButtonVariant = 'primary',
  options: ButtonClassOptions = {},
): string {
  return [
    'btn',
    `btn--${variant}`,
    options.icon ? 'btn--icon' : null,
    options.selected ? 'is-selected' : null,
    options.className,
  ]
    .filter(Boolean)
    .join(' ')
}
