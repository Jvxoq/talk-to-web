export type ButtonVariant = 'primary' | 'secondary' | 'outline' | 'ghost'

interface ButtonClassOptions {
  icon?: boolean
  selected?: boolean
  className?: string
}

/**
 * The one source of button styles, shared by `<Button>` and `motion.button`.
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
