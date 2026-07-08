import { useEffect, useRef, useState, type ChangeEvent, type KeyboardEvent } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL ?? '/generate/text/'
const UPLOAD_URL = import.meta.env.VITE_UPLOAD_URL ?? '/upload/file/'

// Mirrors the Literal options on TextModelRequest.model in schemas.py
const MODELS = ['groq/compound', 'llama-3.1-8b-instant'] as const
type Model = (typeof MODELS)[number]

type Theme = 'light' | 'dark'
const THEME_KEY = 'theme'

interface Message {
  role: 'user' | 'assistant'
  content: string
  error?: boolean
}

interface UploadedFile {
  name: string
  path: string
}

function getInitialTheme(): Theme {
  const stored = localStorage.getItem(THEME_KEY)
  if (stored === 'light' || stored === 'dark') return stored
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [model, setModel] = useState<Model>(MODELS[0])
  const [theme, setTheme] = useState<Theme>(getInitialTheme)
  const [attachedFile, setAttachedFile] = useState<UploadedFile | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem(THEME_KEY, theme)
  }, [theme])

  const scrollToBottom = () => {
    requestAnimationFrame(() => {
      scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight })
    })
  }

  const sendMessage = async () => {
    const text = input.trim()
    if (!text || isStreaming) return

    setInput('')
    setMessages((prev) => [
      ...prev,
      { role: 'user', content: text },
      { role: 'assistant', content: '' },
    ])
    setIsStreaming(true)
    scrollToBottom()

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          user_input: text,
          temperature: 0,
        }),
      })

      if (!response.ok || !response.body) {
        throw new Error(`Request failed with status ${response.status}`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        setMessages((prev) => {
          const next = [...prev]
          next[next.length - 1] = {
            role: 'assistant',
            content: next[next.length - 1].content + chunk,
          }
          return next
        })
        scrollToBottom()
      }
    } catch {
      setMessages((prev) => {
        const next = [...prev]
        next[next.length - 1] = {
          role: 'assistant',
          content: 'Something went wrong. Please try again.',
          error: true,
        }
        return next
      })
    } finally {
      setIsStreaming(false)
    }
  }

  const handleFileSelect = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file) return

    if (file.type !== 'application/pdf') {
      setUploadError('File must be a PDF.')
      return
    }

    setUploadError(null)
    setIsUploading(true)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(UPLOAD_URL, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const body = await response.json().catch(() => null)
        throw new Error(body?.detail ?? `Upload failed with status ${response.status}`)
      }

      const data: { message: string; file_path: string } = await response.json()
      setAttachedFile({ name: file.name, path: data.file_path })
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Failed to upload file.')
    } finally {
      setIsUploading(false)
    }
  }

  const removeAttachedFile = () => {
    setAttachedFile(null)
    setUploadError(null)
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="app">
      <header className="header">
        <svg className="mark" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
          <path d="M12 0 L14 10 L24 12 L14 14 L12 24 L10 14 L0 12 L10 10 Z" />
        </svg>
        <h1>Talk to web</h1>
        <button
          type="button"
          className="theme-toggle"
          onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
          aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {theme === 'dark' ? (
            <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <path d="M12 4V2M12 22v-2M4 12H2m20 0h-2M5.6 5.6 4.2 4.2m15.6 1.4 1.4-1.4M5.6 18.4l-1.4 1.4m15.6-1.4 1.4 1.4M12 7a5 5 0 1 0 0 10 5 5 0 0 0 0-10Z" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <path d="M20.354 15.354A9 9 0 0 1 8.646 3.646 9.003 9.003 0 1 0 20.354 15.354Z" />
            </svg>
          )}
        </button>
      </header>

      <div className="chat" ref={scrollRef}>
        {messages.length === 0 && (
          <div className="empty-state">Ask me anything about a webpage.</div>
        )}
        {messages.map((message, i) => (
          <div key={i} className={`message ${message.role}`}>
            <div className={`bubble ${message.error ? 'error' : ''}`}>
              {message.content ||
                (message.role === 'assistant' && isStreaming && i === messages.length - 1
                  ? '...'
                  : '')}
            </div>
          </div>
        ))}
      </div>

      {(attachedFile || isUploading || uploadError) && (
        <div className="attachment-bar">
          {isUploading && <span className="attachment-status">Uploading…</span>}
          {!isUploading && attachedFile && (
            <span className="attachment-chip">
              {attachedFile.name}
              <button
                type="button"
                className="attachment-remove"
                onClick={removeAttachedFile}
                aria-label="Remove attached file"
              >
                ×
              </button>
            </span>
          )}
          {!isUploading && uploadError && <span className="attachment-error">{uploadError}</span>}
        </div>
      )}

      <div className="composer">
        <input
          ref={fileInputRef}
          type="file"
          accept="application/pdf"
          onChange={handleFileSelect}
          hidden
        />
        <button
          type="button"
          className="upload-btn"
          onClick={() => fileInputRef.current?.click()}
          disabled={isStreaming || isUploading}
          aria-label="Attach a PDF"
        >
          <span aria-hidden="true">📎</span>
        </button>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Message Talk to web..."
          rows={1}
          disabled={isStreaming}
        />
        <select
          className="model-select"
          value={model}
          onChange={(e) => setModel(e.target.value as Model)}
          disabled={isStreaming}
          aria-label="Model"
        >
          {MODELS.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
        <button onClick={sendMessage} disabled={isStreaming || !input.trim()}>
          Send
        </button>
      </div>
    </div>
  )
}

export default App
