import type { ButtonHTMLAttributes, ReactNode, Ref } from 'react'
import { buttonClass, type ButtonVariant } from './buttonVariants'
import './ui.css'

export type IconButtonProps = Omit<
  ButtonHTMLAttributes<HTMLButtonElement>,
  'children'
> & {
  variant?: ButtonVariant
  selected?: boolean
  /** Required — an icon-only control has no accessible name without it. */
  label: string
  icon: ReactNode
  ref?: Ref<HTMLButtonElement>
}

export function IconButton({
  variant = 'secondary',
  selected,
  label,
  icon,
  className,
  ...props
}: IconButtonProps) {
  return (
    <button
      className={buttonClass(variant, { icon: true, selected, className })}
      aria-label={label}
      title={label}
      {...props}
    >
      {icon}
    </button>
  )
}
