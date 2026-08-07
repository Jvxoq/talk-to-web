import type { ButtonHTMLAttributes, Ref } from 'react'
import { buttonClass, type ButtonVariant } from './buttonVariants'
import './ui.css'

export type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant
  selected?: boolean
  loading?: boolean
  ref?: Ref<HTMLButtonElement>
}

export function Button({
  variant = 'primary',
  selected,
  loading,
  className,
  disabled,
  children,
  ...props
}: ButtonProps) {
  return (
    <button
      className={buttonClass(variant, { selected, className })}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      style={loading ? { opacity: 0.6 } : undefined}
      {...props}
    >
      {children}
    </button>
  )
}
