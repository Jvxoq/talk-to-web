import { useState } from 'react'
import { motion } from 'motion/react'
import { LogOut } from 'lucide-react'
import { buttonClass, ErrorBoundary, Mark } from '../components/ui'
import { AuthGate, AuthProvider, useAuth } from '../features/auth'
import {
  ChatLog,
  Composer,
  FALLBACK_DEFAULT_MODEL,
  Sidebar,
  useChat,
  useConversations,
  type Model,
} from '../features/chat'
import { ThemeToggle, useTheme } from '../features/theme'
import type { ConversationOut } from '../lib/conversation'
import { springs } from '../lib/motion'
import '../features/auth/auth.css'
import '../features/chat/chat.css'

/**
 * The chat itself. Rendered only once there is a signed-in user and an active
 * conversation to talk into, which is what lets `useChat` assume every
 * request it makes will be authorised and scoped to a real thread.
 */
function Chat({
  model,
  onModelChange,
  conversationId,
  preloaded,
}: {
  model: Model
  onModelChange: (m: Model) => void
  conversationId: number
  preloaded: ConversationOut | null
}) {
  const { messages, isStreaming, send, cooldownUntil, restoreText, clearRestoreText } = useChat(
    model,
    conversationId,
    preloaded,
  )

  return (
    <main className="main">
      <ChatLog messages={messages} isStreaming={isStreaming} />
      <Composer
        model={model}
        onModelChange={onModelChange}
        onSend={send}
        isStreaming={isStreaming}
        cooldownUntil={cooldownUntil}
        restoreText={restoreText}
        onTextRestored={clearRestoreText}
      />
    </main>
  )
}

function SignOutButton() {
  const { signOut } = useAuth()

  return (
    <motion.button
      type="button"
      className={buttonClass('secondary', { icon: true })}
      onClick={() => void signOut()}
      whileHover={{ x: -2, y: -2 }}
      whileTap={{ x: 0, y: 0 }}
      transition={springs.press}
      aria-label="Sign out"
      title="Sign out"
    >
      <LogOut size={16} />
    </motion.button>
  )
}

function Shell() {
  const [model, setModel] = useState<Model>(FALLBACK_DEFAULT_MODEL)
  const { theme, toggle } = useTheme()
  const { user, status } = useAuth()

  // Idle until signed in - `useConversations` would otherwise fire its first
  // `GET /conversations/` before there is a token to send with it.
  const conversations = useConversations(model, status === 'signed-in')

  return (
    <div className="shell">
      {/* Bypasses `AuthGate` rather than nesting inside it: a second copy of
          the sign-in form next to the main one would be worse than no
          sidebar at all while signed out. */}
      {status === 'signed-in' && (
        <Sidebar
          conversations={conversations.conversations}
          activeId={conversations.activeId}
          isLoading={conversations.isLoading}
          onSelect={conversations.select}
          onNew={() => void conversations.startNew()}
          onDelete={(id) => void conversations.remove(id)}
        />
      )}

      <div className="app">
        <header className="header">
          <Mark
            className="mark"
            initial={{ rotate: -120, scale: 0.5, opacity: 0 }}
            animate={{ rotate: 0, scale: 1, opacity: 1 }}
            transition={{ ...springs.standard, delay: 0.05 }}
          />
          <h1>Talk to web</h1>
          <span className="eyebrow">Beta</span>
          {user && <span className="header__email">{user.email}</span>}
          <ThemeToggle theme={theme} onToggle={toggle} />
          {user && <SignOutButton />}
        </header>

        {/* The gate wraps only the main region: the header stays put across
            sign-in, so the app does not appear to reload around the user. */}
        <AuthGate>
          {conversations.activeId !== null ? (
            <Chat
              model={model}
              onModelChange={setModel}
              conversationId={conversations.activeId}
              preloaded={conversations.preloaded}
            />
          ) : (
            // The conversation list is still loading (or opening a first
            // conversation for a brand new account) - an empty region beats a
            // composer with nowhere to send a message.
            <main className="main" />
          )}
        </AuthGate>
      </div>
    </div>
  )
}

export default function App() {
  return (
    // Outside `AuthProvider`, so a throw from the provider itself - a malformed
    // `/auth/refresh` body on mount, say - still lands on the crash screen
    // rather than on a blank document.
    <ErrorBoundary>
      <AuthProvider>
        <Shell />
      </AuthProvider>
    </ErrorBoundary>
  )
}
