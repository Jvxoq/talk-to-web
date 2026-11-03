import { Fragment, type CSSProperties } from 'react'

type Props = {
  text: string
  as?: 'h1' | 'h2'
  /** Milliseconds between words. The system's reveal stagger is 60ms. */
  stagger?: number
  className?: string
}

/**
 * The hero-title pattern: the line arrives word by word, fading up out of a
 * slight blur. Each word carries its delay as a CSS variable; the animation
 * itself lives in globals.css behind `prefers-reduced-motion: no-preference`,
 * and `aria-label` + `aria-hidden` keep the split words reading as one phrase.
 *
 * The separator space is a text node BETWEEN the spans, not inside one: a
 * trailing space inside an `inline-block` collapses, and the words would jam
 * together exactly while the reveal animation is on.
 */
export function WordReveal({ text, as: Tag = 'h1', stagger = 60, className }: Props) {
  const words = text.split(' ')
  return (
    <Tag className={className} aria-label={text}>
      {words.map((word, i) => (
        <Fragment key={`${word}-${i}`}>
          <span
            aria-hidden="true"
            className="word"
            style={{ '--delay': `${i * stagger}ms` } as CSSProperties}
          >
            {word}
          </span>
          {i < words.length - 1 ? ' ' : null}
        </Fragment>
      ))}
    </Tag>
  )
}
