import { useEffect, useRef } from 'react'
import { AnimatePresence } from 'motion/react'
import type { Message } from '../types'
import { EmptyState } from './EmptyState'
import { MessageBubble } from './MessageBubble'

interface ChatLogProps {
  messages: Message[]
  isStreaming: boolean
}

export function ChatLog({ messages, isStreaming }: ChatLogProps) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const lastIndex = messages.length - 1

  // Scrolling is an imperative DOM concern, so it belongs in an effect —
  // re-running on content change keeps the newest token in view as it streams.
  useEffect(() => {
    const node = scrollRef.current
    if (!node) return
    const frame = requestAnimationFrame(() =>
      node.scrollTo({ top: node.scrollHeight }),
    )
    return () => cancelAnimationFrame(frame)
  }, [messages])

  return (
    // role="log" is already a polite live region. No aria-live, or every token is announced.
    <div className="chat" ref={scrollRef} role="log">
      {messages.length === 0 && <EmptyState />}

      <AnimatePresence initial={false}>
        {messages.map((message, i) => (
          <MessageBubble
            key={message.id}
            message={message}
            pending={isStreaming && i === lastIndex}
          />
        ))}
      </AnimatePresence>
    </div>
  )
}
