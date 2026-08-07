import { memo } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { springs } from '../../../lib/motion'
import { normalizeStreamingMarkdown } from '../markdown'
import type { Message, ToolActivity } from '../types'
import { TypingDots } from './TypingDots'

interface MessageBubbleProps {
  message: Message
  /** The assistant turn is still filling in — render the thinking state. */
  pending: boolean
}

/** `search_web` -> `Search web`. Used to build a label when there's no `summary`. */
function toolLabel(name: string): string {
  return name
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}

/**
 * The chip's text carries the state, not just its color — a screen reader (or
 * a colorblind reader) gets "failed" or "done" from the words alone.
 */
function toolChipText(tool: ToolActivity): string {
  switch (tool.status) {
    case 'start':
      return tool.summary ?? `${toolLabel(tool.name)}…`
    case 'ok':
      return `${toolLabel(tool.name)} done`
    case 'failed':
      return `${toolLabel(tool.name)} failed`
  }
}

// Memoised because every delta re-renders the whole transcript, and reparsing
// the markdown of settled turns on each token is the expensive part of that.
// `tools` fits this untouched: useChat only ever replaces `message` with a
// fresh object (never mutates it in place), so the default shallow prop
// comparison still short-circuits every render that didn't touch this turn.
export const MessageBubble = memo(function MessageBubble({
  message,
  pending,
}: MessageBubbleProps) {
  const reduce = useReducedMotion()
  const isUser = message.role === 'user'
  const renderMarkdown = message.role === 'assistant' && !message.error
  const markdownSource = pending
    ? normalizeStreamingMarkdown(message.content)
    : message.content

  return (
    <motion.div
      className={`message ${message.role}`}
      initial={{
        opacity: 0,
        y: reduce ? 0 : 12,
        x: reduce ? 0 : isUser ? 12 : -12,
      }}
      animate={{ opacity: 1, y: 0, x: 0 }}
      transition={springs.card}
    >
      <span className="message-role">{isUser ? 'You' : 'Assistant'}</span>
      {message.tools && message.tools.length > 0 && (
        // Polite, and separate from the transcript's own (deliberately
        // unannounced) log region — tool chips change a handful of times per
        // turn, not once per token, so announcing them doesn't spam.
        <div className="tool-chips" aria-live="polite">
          <AnimatePresence initial={false}>
            {message.tools.map((tool) => (
              <motion.span
                key={tool.name}
                className={`tool-chip is-${tool.status}`}
                initial={{ opacity: 0, y: reduce ? 0 : 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: reduce ? 0 : -6 }}
                transition={springs.card}
              >
                {toolChipText(tool)}
              </motion.span>
            ))}
          </AnimatePresence>
        </div>
      )}
      <div className={`bubble${message.error ? ' error' : ''}`}>
        {message.content ? (
          renderMarkdown ? (
            <div className="markdown">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{markdownSource}</ReactMarkdown>
            </div>
          ) : (
            message.content
          )
        ) : pending ? (
          <TypingDots />
        ) : null}
      </div>
    </motion.div>
  )
})
