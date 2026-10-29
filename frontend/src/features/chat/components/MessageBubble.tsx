import { memo } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { ExternalLink } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { springs } from '../../../lib/motion'
import { normalizeStreamingMarkdown } from '../markdown'
import { summarizingText } from '../summarizing'
import type { Message, Source, ToolActivity } from '../types'
import { formatDuration, formatTokens } from '../usage'
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
 * The chip's text states the status, not just its color, for accessibility.
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

/**
 * Every source behind this turn's tool calls, deduplicated by URL or label.
 */
function collectSources(tools: ToolActivity[] | undefined): Source[] {
  if (!tools) return []
  const seen = new Map<string, Source>()
  for (const tool of tools) {
    for (const source of tool.sources ?? []) {
      seen.set(source.url ?? source.label, source)
    }
  }
  return [...seen.values()]
}

// Memoised, so settled turns do not reparse markdown on every token. `useChat`
// replaces `message` objects, never mutates them, so shallow compare works.
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
  const sources = message.role === 'assistant' ? collectSources(message.tools) : []
  const summarizing = message.role === 'assistant' ? message.summarizing : undefined
  const hasChips = summarizing !== undefined || (message.tools?.length ?? 0) > 0

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
      {hasChips && (
        // Announced politely. Chips change rarely, so this does not spam.
        <div className="tool-chips" aria-live="polite">
          <AnimatePresence initial={false}>
            {/* First in the row, because it happens before any tool call this
                turn makes and reads as the reason the reply has not started. */}
            {summarizing && (
              <motion.span
                key="summarizing"
                className={`tool-chip is-${summarizing.status === 'start' ? 'start' : 'ok'}`}
                initial={{ opacity: 0, y: reduce ? 0 : 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: reduce ? 0 : -6 }}
                transition={springs.card}
              >
                {summarizingText(summarizing)}
              </motion.span>
            )}
            {(message.tools ?? []).map((tool) => (
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
      {/* Announced once when it lands: the transcript is a `role="log"`, which
          a screen reader reads politely and may not surface at all mid-stream,
          and a refusal is the one turn the user has to act on. */}
      <div
        className={`bubble${message.error ? ' error' : ''}`}
        role={message.error ? 'alert' : undefined}
      >
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
      {sources.length > 0 && (
        // Shows what the answer was based on.
        <ul className="source-list" aria-label="Sources">
          {sources.map((source) => (
            <li key={source.url ?? source.label}>
              {source.url ? (
                <a
                  className="source-chip"
                  href={source.url}
                  target="_blank"
                  rel="noreferrer"
                  title={source.label}
                >
                  <span className="source-chip-label">{source.label}</span>
                  <ExternalLink strokeWidth={2} aria-hidden="true" />
                </a>
              ) : (
                <span className="source-chip" title={source.label}>
                  <span className="source-chip-label">{source.label}</span>
                </span>
              )}
            </li>
          ))}
        </ul>
      )}
      {message.usage && !pending && (
        // Plain text, so it reads as a footnote.
        <span className="usage-note" title={message.usage.model}>
          {formatTokens(message.usage)} · {formatDuration(message.usage)}
        </span>
      )}
    </motion.div>
  )
})
