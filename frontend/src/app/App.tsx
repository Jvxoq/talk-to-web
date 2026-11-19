import { useState } from 'react'
import { createPortal } from 'react-dom'
import { motion, useReducedMotion } from 'motion/react'
import { LogOut } from 'lucide-react'
import { buttonClass, ErrorBoundary, Mark } from '../components/ui'
import { AuthGate, AuthProvider, useAuth } from '../features/auth'
import {
  ChatLog,
  Composer,
  ConversationSkeleton,
  FALLBACK_DEFAULT_MODEL,
  Sidebar,
  useChat,
  useConversations,
  type Model,
} from '../features/chat'
import { ThemeToggle, useTheme } from '../features/theme'
import type { ConversationOut } from '../lib/conversation'
import { springs } from '../lib/motion'
import { BackendGate } from './BackendGate'
import './app.css'
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
  const {
    messages,
    isLoading,
    isStreaming,
    send,
    cooldownUntil,
    restoreText,
    clearRestoreText,
  } = useChat(model, conversationId, preloaded)

  return (
    <main className="main">
      <ChatLog messages={messages} isLoading={isLoading} isStreaming={isStreaming} />
      <Composer
        model={model}
        conversationId={conversationId}
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
  const { signOut, user } = useAuth()
  // The address rides on the label too: on a phone the sidebar strip has no
  // room for it, and this is the one control tied to the account.
  const label = user ? `Sign out of ${user.email}` : 'Sign out'

  return (
    <motion.button
      type="button"
      className={buttonClass('secondary', { icon: true })}
      onClick={() => void signOut()}
      aria-label={label}
      title={label}
    >
      <LogOut size={16} strokeWidth={1.5} aria-hidden="true" />
    </motion.button>
  )
}

function Shell({ cornerSlot }: { cornerSlot: HTMLElement | null }) {
  const [model, setModel] = useState<Model>(FALLBACK_DEFAULT_MODEL)
  const { user, status } = useAuth()
  const reduce = useReducedMotion()

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
          error={conversations.error}
          account={user?.email}
          onDelete={(id) => void conversations.remove(id)}
        />
      )}

      <div className="app">
        {/* The brand bar shows only while signed out. The chat page gives the
            whole column to the conversation, and keeps the h1 for screen
            readers. */}
        {status === 'signed-in' ? (
          <h1 className="sr-only">Talk to web</h1>
        ) : (
          <header className="header">
            <Mark
              className="mark"
              initial={{ opacity: 0, scale: reduce ? 1 : 0.85 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ ...springs.standard, delay: 0.05 }}
            />
            <h1>Talk to web</h1>
          </header>
        )}

        {/* Sign out joins the theme toggle in the top-right corner. The corner
            lives above the backend gate, so it reaches it through a portal. */}
        {user && cornerSlot && createPortal(<SignOutButton />, cornerSlot)}

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
            // conversation for a brand new account). Same wait the transcript
            // itself shows, held here as one shape the two share - an empty
            // region, and then a composer with nowhere to send a message.
            <main className="main">
              <div className="chat">
                <ConversationSkeleton />
              </div>
            </main>
          )}
        </AuthGate>
      </div>
    </div>
  )
}

export default function App() {
  const { theme, toggle } = useTheme()
  const [cornerSlot, setCornerSlot] = useState<HTMLElement | null>(null)

  return (
    // Outside `AuthProvider`, so a throw from the provider itself - a malformed
    // `/auth/refresh` body on mount, say - still lands on the crash screen
    // rather than on a blank document.
    <ErrorBoundary>
      {/* Above `AuthProvider`, because that provider's mount effect is the
          first backend call the app makes. The free instance sleeps, and a
          failed `/auth/refresh` looks exactly like being signed out - so the
          wait happens here, before anything asks. */}
      <BackendGate>
        <AuthProvider>
          <Shell cornerSlot={cornerSlot} />
        </AuthProvider>
      </BackendGate>

      {/* Top-right on every screen, the backend wait included: it sits outside
          the gate so the theme can change while the server wakes. Last in the
          DOM so keyboard users reach the page's own content first; it is fixed,
          so the order does not move it on screen. */}
      <div className="corner">
        <ThemeToggle theme={theme} onToggle={toggle} />
        <span ref={setCornerSlot} className="corner__slot" />
      </div>
    </ErrorBoundary>
  )
}
