export type ButtonVariant = 'primary' | 'secondary' | 'outline' | 'ghost'

interface ButtonClassOptions {
  icon?: boolean
  selected?: boolean
  className?: string
}

/**
 * The single source of truth for button styling.
 *
 * Lives apart from the component because the interactive buttons in this app
 * are Motion components (`motion.button`) so they can carry springs — they take
 * the class from here rather than wrapping `<Button>`, which keeps one variant
 * table for both paths.
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
