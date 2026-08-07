import { useState } from 'react'
import { Mark } from '../components/ui/Mark'
import { ChatLog, Composer, FALLBACK_DEFAULT_MODEL, useChat, type Model } from '../features/chat'
import { ThemeToggle, useTheme } from '../features/theme'
import { springs } from '../lib/motion'
import '../features/chat/chat.css'

export default function App() {
  const [model, setModel] = useState<Model>(FALLBACK_DEFAULT_MODEL)
  const { messages, isStreaming, send } = useChat(model)
  const { theme, toggle } = useTheme()

  return (
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
        <ThemeToggle theme={theme} onToggle={toggle} />
      </header>

      <main className="main">
        <ChatLog messages={messages} isStreaming={isStreaming} />
        <Composer
          model={model}
          onModelChange={setModel}
          onSend={send}
          isStreaming={isStreaming}
        />
      </main>
    </div>
  )
}
