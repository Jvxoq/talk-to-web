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
import { useCountdown } from '../hooks/useCountdown'
import { useModels } from '../hooks/useModels'
import { isVoiceInputSupported, useVoiceInput } from '../hooks/useVoiceInput'
import type { Model } from '../types'

interface ComposerProps {
  model: Model
  /** The thread an attached file belongs to. Every upload names it. */
  conversationId: number
  onModelChange: (model: Model) => void
  onSend: (text: string) => void
  isStreaming: boolean
  /** Epoch ms until which sending is refused by the server, or null. */
  cooldownUntil: number | null
  /** Text handed back after a refused send, so the user does not lose it. */
  restoreText: string | null
  onTextRestored: () => void
}

const stripMotion = {
  initial: { opacity: 0, height: 0 },
  animate: { opacity: 1, height: 'auto' },
  exit: { opacity: 0, height: 0 },
  transition: { duration: timing.standard, ease: easeStandard },
  style: { overflow: 'hidden' as const },
}

export function Composer({
  model,
  conversationId,
  onModelChange,
  onSend,
  isStreaming,
  cooldownUntil,
  restoreText,
  onTextRestored,
}: ComposerProps) {
  const [input, setInput] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)
  const reduce = useReducedMotion()

  const { models, defaultModel } = useModels()
  const upload = useFileUpload(conversationId)

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

  const sendWait = useCountdown(cooldownUntil)
  const uploadWait = useCountdown(upload.cooldownUntil)
  const canSend = !isStreaming && sendWait === 0 && input.trim().length > 0

  // A refused send hands the text back rather than swallowing it: the limits
  // are tight enough to meet in normal use, and losing a typed message to one
  // would be the app's fault, not the user's.
  useEffect(() => {
    if (restoreText === null) return
    setInput(restoreText)
    onTextRestored()
  }, [restoreText, onTextRestored])

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

  const showAttachmentStrip = Boolean(
    upload.file || upload.isUploading || upload.error || sendWait > 0,
  )
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
                    onClick={() => void upload.clear()}
                    aria-label="Remove attached file"
                  >
                    <X strokeWidth={2} aria-hidden="true" />
                  </button>
                </span>
              )}

              {!upload.isUploading && upload.error && (
                <span className="status-label is-error">{upload.error}</span>
              )}

              {/* `role="status"` rather than `alert`: this updates every second,
                  and an assertive region would interrupt a screen reader on
                  every tick instead of announcing the limit once. */}
              {sendWait > 0 && (
                <span className="status-label is-error" role="status">
                  Message limit reached. You can send again in {sendWait}s
                </span>
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
          accept=".pdf,.txt,.md,.docx,application/pdf,text/plain,text/markdown,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
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
              disabled={isStreaming || upload.isUploading || uploadWait > 0}
              whileHover={uploadWait > 0 ? undefined : { x: -2, y: -2 }}
              whileTap={{ x: 0, y: 0 }}
              transition={springs.press}
              aria-label={
                uploadWait > 0 ? `Upload limit reached, wait ${uploadWait} seconds` : 'Attach a PDF'
              }
              title={
                uploadWait > 0
                  ? `Upload limit reached. Try again in ${uploadWait}s`
                  : 'Attach a PDF'
              }
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
              aria-label={
                sendWait > 0 ? `Message limit reached, wait ${sendWait} seconds` : 'Send message'
              }
              title={sendWait > 0 ? `Try again in ${sendWait}s` : undefined}
            >
              {/* The seconds replace the arrow rather than sitting beside it:
                  the button is one square, and the number is the only thing
                  worth reading while the wait is live. */}
              {sendWait > 0 ? (
                <span className="send-countdown">{sendWait}</span>
              ) : (
                <ArrowUp strokeWidth={2.5} aria-hidden="true" />
              )}
            </motion.button>
          </div>
        </div>
      </div>
    </div>
  )
}
