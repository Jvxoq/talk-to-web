import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { Plus, Trash2 } from 'lucide-react'
import { buttonClass, IconButton } from '../../../components/ui'
import { springs } from '../../../lib/motion'
import type { ConversationSummary } from '../../../lib/conversation'

interface SidebarProps {
  conversations: ConversationSummary[]
  activeId: number | null
  isLoading: boolean
  onSelect: (id: number) => void
  onNew: () => void
  onDelete: (id: number) => void
}

/**
 * The conversation list, new-chat control and per-row delete.
 *
 * `deleteConversation` in `lib/conversation.ts` existed before this component
 * did and was never called from anywhere - this is the UI that finally calls
 * it.
 */
export function Sidebar({ conversations, activeId, isLoading, onSelect, onNew, onDelete }: SidebarProps) {
  const reduce = useReducedMotion() ?? false

  return (
    <nav className="sidebar" aria-label="Conversations">
      <div className="sidebar-header">
        <span className="sidebar-title">Chats</span>
        <IconButton
          variant="secondary"
          label="New chat"
          icon={<Plus strokeWidth={2} aria-hidden="true" />}
          onClick={() => void onNew()}
        />
      </div>

      <ul className="sidebar-list">
        {conversations.length === 0 && (
          <li className="sidebar-empty">{isLoading ? 'Loading…' : 'No conversations yet'}</li>
        )}

        <AnimatePresence initial={false}>
          {conversations.map((conversation) => {
            const selected = conversation.id === activeId
            return (
              <motion.li
                key={conversation.id}
                className={`sidebar-item${selected ? ' is-selected' : ''}`}
                initial={{ opacity: 0, y: reduce ? 0 : -8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={springs.card}
              >
                <button
                  type="button"
                  className="sidebar-item-select"
                  onClick={() => onSelect(conversation.id)}
                  aria-current={selected ? 'true' : undefined}
                >
                  <span className="sidebar-item-title">{conversation.title}</span>
                </button>
                <motion.button
                  type="button"
                  className={buttonClass('ghost', { icon: true, className: 'sidebar-item-delete' })}
                  onClick={() => void onDelete(conversation.id)}
                  aria-label={`Delete ${conversation.title}`}
                  title="Delete conversation"
                  whileHover={{ x: -1, y: -1 }}
                  whileTap={{ x: 0, y: 0 }}
                  transition={springs.press}
                >
                  <Trash2 strokeWidth={2} aria-hidden="true" />
                </motion.button>
              </motion.li>
            )
          })}
        </AnimatePresence>
      </ul>
    </nav>
  )
}
