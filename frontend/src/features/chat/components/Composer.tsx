import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type KeyboardEvent,
} from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { ArrowUp, Mic, Paperclip, X } from 'lucide-react'
import { buttonClass } from '../../../components/ui'
import { easeStandard, springs, timing } from '../../../lib/motion'
import { useFileUpload } from '../hooks/useFileUpload'
import { useModels } from '../hooks/useModels'
import { isVoiceInputSupported, useVoiceInput } from '../hooks/useVoiceInput'
import type { Model } from '../types'

interface ComposerProps {
  model: Model
  onModelChange: (model: Model) => void
  onSend: (text: string) => void
  isStreaming: boolean
}

const stripMotion = {
  initial: { opacity: 0, height: 0 },
  animate: { opacity: 1, height: 'auto' },
  exit: { opacity: 0, height: 0 },
  transition: { duration: timing.standard, ease: easeStandard },
  style: { overflow: 'hidden' as const },
}

export function Composer({ model, onModelChange, onSend, isStreaming }: ComposerProps) {
  const [input, setInput] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)
  const reduce = useReducedMotion()

  const { models, defaultModel } = useModels()
  const upload = useFileUpload()

  // The caller seeds `model` with a hardcoded fallback before this list can
  // possibly have loaded. Once the real list lands, swap to its default if
  // the current value isn't (or is no longer) one of the real options.
  useEffect(() => {
    if (models.length > 0 && !models.includes(model)) {
      onModelChange(defaultModel)
    }
  }, [models, defaultModel, model, onModelChange])

  // Each finalised utterance is appended to whatever is already in the composer,
  // so dictation adds to typed text rather than replacing it.
  const appendTranscript = useCallback((text: string) => {
    setInput((prev) => (prev ? `${prev.trimEnd()} ${text}` : text))
  }, [])

  const voice = useVoiceInput(appendTranscript)
  const isRecording = voice.status !== 'idle'
  const voiceSupported = isVoiceInputSupported()
  const canSend = !isStreaming && input.trim().length > 0

  const submit = () => {
    if (!canSend) return
    if (isRecording) voice.stop()
    onSend(input.trim())
    setInput('')
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const handleFileSelect = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (file) void upload.upload(file)
  }

  const showAttachmentStrip = Boolean(upload.file || upload.isUploading || upload.error)
  const showVoiceStrip = isRecording || Boolean(voice.error)

  return (
    <div className="composer-wrap">
      <div className="composer">
        <AnimatePresence initial={false}>
          {showAttachmentStrip && (
            <motion.div className="composer-strip" {...stripMotion}>
              {upload.isUploading && <span className="status-label">Uploading…</span>}

              {!upload.isUploading && upload.file && (
                <span className="attachment-chip">
                  <span className="attachment-name">{upload.file.name}</span>
                  <button
                    type="button"
                    className="attachment-remove"
                    onClick={upload.clear}
                    aria-label="Remove attached file"
                  >
                    <X strokeWidth={2} aria-hidden="true" />
                  </button>
                </span>
              )}

              {!upload.isUploading && upload.error && (
                <span className="status-label is-error">{upload.error}</span>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence initial={false}>
          {showVoiceStrip && (
            <motion.div className="composer-strip" {...stripMotion}>
              {voice.status === 'connecting' && (
                <span className="status-label">Connecting…</span>
              )}

              {voice.status === 'listening' && (
                <>
                  <motion.span
                    className="voice-dot"
                    aria-hidden="true"
                    animate={reduce ? undefined : { opacity: [1, 0.3, 1] }}
                    transition={{ duration: 1.8, repeat: Infinity, ease: 'easeInOut' }}
                  />
                  <span className="status-label">Listening…</span>
                  {voice.interim && <span className="voice-interim">{voice.interim}</span>}
                </>
              )}

              {voice.error && <span className="status-label is-error">{voice.error}</span>}
            </motion.div>
          )}
        </AnimatePresence>

        <input
          ref={fileInputRef}
          type="file"
          accept="application/pdf"
          onChange={handleFileSelect}
          hidden
        />

        <label className="sr-only" htmlFor="composer-input">
          Message
        </label>
        <textarea
          id="composer-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask anything…"
          rows={1}
          disabled={isStreaming}
        />

        <div className="composer-controls">
          <div className="controls-left">
            <motion.button
              type="button"
              className={buttonClass('secondary', { icon: true })}
              onClick={() => fileInputRef.current?.click()}
              disabled={isStreaming || upload.isUploading}
              whileHover={{ x: -2, y: -2 }}
              whileTap={{ x: 0, y: 0 }}
              transition={springs.press}
              aria-label="Attach a PDF"
              title="Attach a PDF"
            >
              <Paperclip strokeWidth={2} aria-hidden="true" />
            </motion.button>

            <span className="mic-wrap">
              <AnimatePresence>
                {isRecording && !reduce && (
                  <motion.span
                    className="mic-ring"
                    aria-hidden="true"
                    initial={{ scale: 1, opacity: 0.5 }}
                    animate={{ scale: 1.3, opacity: 0 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 1.6, repeat: Infinity, ease: 'easeOut' }}
                  />
                )}
              </AnimatePresence>

              <motion.button
                type="button"
                className={buttonClass('secondary', {
                  selected: isRecording,
                  className: 'talk-btn',
                })}
                onClick={voice.toggle}
                disabled={isStreaming || !voiceSupported}
                whileHover={{ x: -2, y: -2 }}
                whileTap={{ x: 0, y: 0 }}
                transition={springs.press}
                aria-pressed={isRecording}
                title={
                  voiceSupported
                    ? isRecording
                      ? 'Stop talking'
                      : 'Talk to type'
                    : "Voice input isn't supported in this browser"
                }
              >
                <Mic strokeWidth={2} aria-hidden="true" />
                <span>{isRecording ? 'Stop' : 'Talk'}</span>
              </motion.button>
            </span>
          </div>

          <div className="controls-right">
            <select
              className="model-select"
              value={model}
              onChange={(e) => onModelChange(e.target.value as Model)}
              disabled={isStreaming}
              aria-label="Model"
            >
              {models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>

            <motion.button
              type="button"
              className={buttonClass('primary', { icon: true })}
              onClick={submit}
              disabled={!canSend}
              whileHover={canSend ? { x: -2, y: -2 } : undefined}
              whileTap={{ x: 0, y: 0 }}
              transition={springs.press}
              aria-label="Send message"
            >
              <ArrowUp strokeWidth={2.5} aria-hidden="true" />
            </motion.button>
          </div>
        </div>
      </div>
    </div>
  )
}
