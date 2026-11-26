import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ChangeEvent,
  type KeyboardEvent,
} from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { ArrowUp, Mic, Paperclip, X } from 'lucide-react'
import { buttonClass } from '../../../components/ui'
import { easings, timing } from '../../../lib/motion'
import { useFileUpload, type Attachment } from '../hooks/useFileUpload'
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
  transition: { duration: timing.height, ease: easings.standard },
  style: { overflow: 'hidden' as const },
}

// "Uploading" covers both the request and the indexing behind it: the file is
// not usable until both are done, so the difference is not the user's concern.
const ATTACHMENT_LABEL: Record<Attachment['status'], string> = {
  uploading: 'Uploading…',
  indexing: 'Uploading…',
  ready: 'Upload complete',
  failed: 'Upload failed',
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
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Grow the box with its text, up to the CSS max-height, then scroll.
  // Runs on every change to `input`, so typed, dictated and restored text
  // all resize it, and clearing it after a send shrinks it back to one line.
  useLayoutEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${el.scrollHeight}px`
  }, [input])
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
  // Capturing drives every looping indicator: the mic ring, the voice dot,
  // and the selected Talk style. `stopping` is the flush wait after Stop —
  // the mic is already released, so all loops end at once.
  const isRecording = voice.status === 'connecting' || voice.status === 'listening'
  const isFinishing = voice.status === 'stopping'
  const voiceSupported = isVoiceInputSupported()
  const talkDisabled = isStreaming || !voiceSupported || isFinishing

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

  const showAttachmentStrip = Boolean(upload.attachment || upload.error || sendWait > 0)
  const showVoiceStrip = isRecording || isFinishing || Boolean(voice.error)
  const attachment = upload.attachment

  return (
    <div className="composer-wrap">
      <div className="composer">
        <AnimatePresence initial={false}>
          {showAttachmentStrip && (
            <motion.div className="composer-strip" {...stripMotion}>
              {attachment && (
                <span className={`attachment-chip is-${attachment.status}`} role="status">
                  <span className="attachment-name">{attachment.name}</span>
                  <span className="attachment-status">{ATTACHMENT_LABEL[attachment.status]}</span>
                  {attachment.status === 'failed' && attachment.reason && (
                    <span className="attachment-reason">{attachment.reason}</span>
                  )}
                  {attachment.status === 'ready' && (
                    <button
                      type="button"
                      className="attachment-remove"
                      onClick={() => void upload.clear()}
                      aria-label="Remove attached file"
                    >
                      <X strokeWidth={1.5} aria-hidden="true" />
                    </button>
                  )}
                </span>
              )}

              {upload.error && <span className="status-label is-error">{upload.error}</span>}

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

              {voice.status === 'stopping' && (
                <span className="status-label">Finishing…</span>
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
          ref={textareaRef}
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
            <button
              type="button"
              className={buttonClass('secondary', { icon: true })}
              onClick={() => fileInputRef.current?.click()}
              disabled={isStreaming || upload.isUploading || uploadWait > 0}
              aria-label={
                uploadWait > 0 ? `Upload limit reached, wait ${uploadWait} seconds` : 'Attach a PDF'
              }
              title={
                uploadWait > 0
                  ? `Upload limit reached. Try again in ${uploadWait}s`
                  : 'Attach a PDF'
              }
            >
              <Paperclip strokeWidth={1.5} aria-hidden="true" />
            </button>

            <span className="mic-wrap">
              <AnimatePresence>
                {isRecording && !reduce && (
                  <motion.span
                    className="mic-ring"
                    aria-hidden="true"
                    // Each loop starts and ends at opacity 0, so the restart
                    // is invisible. Starting at 0.5 made the ring flash back
                    // in at full strength every 1.6s.
                    initial={{ scale: 1, opacity: 0 }}
                    animate={{ scale: [1, 1.15], opacity: [0, 0.45, 0] }}
                    // Own exit transition: without it the exit inherits the
                    // infinite repeat below, never finishes, and the ring
                    // stays on screen after the user clicks Stop.
                    exit={{
                      opacity: 0,
                      transition: { duration: timing.base, ease: easings.inQuart },
                    }}
                    transition={{ duration: 1.6, repeat: Infinity, ease: 'easeOut' }}
                  />
                )}
              </AnimatePresence>

              <button
                type="button"
                className={buttonClass('secondary', {
                  selected: isRecording,
                  className: 'talk-btn',
                })}
                onClick={voice.toggle}
                disabled={talkDisabled}
                aria-pressed={isRecording}
                title={
                  !voiceSupported
                    ? "Voice input isn't supported in this browser"
                    : isFinishing
                      ? 'Finishing…'
                      : isRecording
                        ? 'Stop talking'
                        : 'Talk to type'
                }
              >
                <Mic strokeWidth={1.5} aria-hidden="true" />
                <span>{isRecording ? 'Stop' : 'Talk'}</span>
              </button>
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

            <button
              type="button"
              className={buttonClass('primary', { icon: true })}
              onClick={submit}
              disabled={!canSend}
              aria-label={
                sendWait > 0 ? `Message limit reached, wait ${sendWait} seconds` : 'Send message'
              }
              title={sendWait > 0 ? `Try again in ${sendWait}s` : undefined}
            >
              {/* The seconds replace the arrow rather than sitting beside it:
                  the button is one circle, and the number is the only thing
                  worth reading while the wait is live. */}
              {sendWait > 0 ? (
                <span className="send-countdown">{sendWait}</span>
              ) : (
                <ArrowUp strokeWidth={1.5} aria-hidden="true" />
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
