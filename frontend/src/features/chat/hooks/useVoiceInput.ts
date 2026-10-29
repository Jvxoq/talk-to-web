import { useCallback, useEffect, useRef, useState } from 'react'
import { getAccessToken } from '../../../lib/session'

export type VoiceStatus = 'idle' | 'connecting' | 'listening' | 'stopping'

const WS_PATH = '/ws/transcribe/'
// Deepgram's recommended speech rate. The browser resamples to it.
const TARGET_SAMPLE_RATE = 16000

// The fixed first subprotocol, before the token. See `socketUrl`.
const TOKEN_SUBPROTOCOL = 'access_token'

interface ServerMessage {
  type: 'ready' | 'transcript' | 'done' | 'error'
  text?: string
  is_final?: boolean
  speech_final?: boolean
  detail?: string
}

/** Everything that has to be torn down when a session ends. */
interface Session {
  socket: WebSocket
  context: AudioContext
  stream: MediaStream
  node: AudioWorkletNode
  /** True once `stop()` has fired. Guards double-send and double-release. */
  stopped: boolean
  /** True once the mic and audio graph are released. Guards a second release. */
  audioReleased: boolean
}

/**
 * Where to open the transcription socket.
 *
 * Vercel's proxy cannot carry a WebSocket upgrade, so `VITE_WS_URL` points
 * straight at the backend. Unset, it uses same-origin, which works with the
 * Vite dev proxy. This page's origin must be in ALLOWED_WEBSOCKET_ORIGINS.
 *
 * The token goes in the subprotocols, not the URL, so it stays out of logs.
 * The backend accepts with `TOKEN_SUBPROTOCOL`, never echoing the token.
 */
function socketUrl(): string {
  return (
    import.meta.env.VITE_WS_URL ??
    `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}${WS_PATH}`
  )
}

export const isVoiceInputSupported = () =>
  typeof window !== 'undefined' &&
  !!navigator.mediaDevices?.getUserMedia &&
  typeof window.AudioWorkletNode !== 'undefined'

/**
 * Streams mic audio to the backend and returns transcripts.
 *
 * Interim results go in `interim` as a preview. Final results go to
 * `onFinalTranscript`.
 */
export function useVoiceInput(onFinalTranscript: (text: string) => void) {
  const [status, setStatus] = useState<VoiceStatus>('idle')
  const [interim, setInterim] = useState('')
  const [error, setError] = useState<string | null>(null)

  const sessionRef = useRef<Session | null>(null)
  // Kept in a ref so the socket handler never closes over a stale callback.
  const onFinalRef = useRef(onFinalTranscript)
  useEffect(() => {
    onFinalRef.current = onFinalTranscript
  }, [onFinalTranscript])

  /** Releases the mic and audio graph. The socket is closed separately, since
   *  we keep it open a moment longer to receive the final transcript. */
  const releaseAudio = useCallback(() => {
    const session = sessionRef.current
    // Called twice on stop, so it must be idempotent.
    if (!session || session.audioReleased) return
    session.audioReleased = true
    session.node.port.onmessage = null
    session.node.disconnect()
    session.stream.getTracks().forEach((track) => track.stop())
    void session.context.close()
  }, [])

  const teardown = useCallback(() => {
    const session = sessionRef.current
    if (!session) return
    releaseAudio()
    session.socket.close()
    sessionRef.current = null
    setStatus('idle')
    setInterim('')
  }, [releaseAudio])

  const start = useCallback(async () => {
    if (sessionRef.current) return

    // Check the token before asking for mic permission.
    const token = getAccessToken()
    if (token === null) {
      setError('Your session has expired. Please sign in again.')
      return
    }

    setError(null)
    setInterim('')
    setStatus('connecting')

    let stream: MediaStream
    let context: AudioContext
    let node: AudioWorkletNode

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      })

      // The browser resamples, so the worklet only converts float32 to int16.
      context = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE })
      await context.audioWorklet.addModule('/pcm-worklet.js')
      node = new AudioWorkletNode(context, 'pcm-encoder')

      // A worklet only runs if connected, so route it through a muted gain.
      const mute = context.createGain()
      mute.gain.value = 0
      context.createMediaStreamSource(stream).connect(node)
      node.connect(mute).connect(context.destination)
    } catch (err) {
      setStatus('idle')
      setError(
        err instanceof DOMException && err.name === 'NotAllowedError'
          ? 'Microphone access was denied.'
          : 'Could not start the microphone.',
      )
      return
    }

    const socket = new WebSocket(socketUrl(), [TOKEN_SUBPROTOCOL, token])
    socket.binaryType = 'arraybuffer'
    sessionRef.current = { socket, context, stream, node, stopped: false, audioReleased: false }

    socket.onopen = () => {
      // Send the actual rate, since browsers may not honour the request.
      socket.send(
        JSON.stringify({ type: 'start', sample_rate: context.sampleRate }),
      )
      node.port.onmessage = (event: MessageEvent<ArrayBuffer>) => {
        if (socket.readyState === WebSocket.OPEN) socket.send(event.data)
      }
    }

    socket.onmessage = (event) => {
      const message: ServerMessage = JSON.parse(event.data)

      switch (message.type) {
        case 'ready':
          setStatus('listening')
          break
        case 'transcript':
          if (message.is_final) {
            // Committed text: hand it to the composer and clear the preview.
            if (message.text) onFinalRef.current(message.text)
            setInterim('')
          } else {
            setInterim(message.text ?? '')
          }
          break
        case 'done':
          teardown()
          break
        case 'error':
          setError(message.detail ?? 'Transcription failed.')
          teardown()
          break
      }
    }

    socket.onerror = () => {
      setError('Lost connection to the transcription service.')
      teardown()
    }

    socket.onclose = () => {
      if (sessionRef.current) teardown()
    }
  }, [teardown])

  const stop = useCallback(() => {
    const session = sessionRef.current
    if (!session || session.stopped) return
    session.stopped = true

    // Release the mic now, but keep the socket open for the final transcript.
    // The server sends "done", then `teardown()` returns to `idle`.
    releaseAudio()
    setStatus('stopping')
    if (session.socket.readyState === WebSocket.OPEN) {
      session.socket.send(JSON.stringify({ type: 'stop' }))
    } else {
      teardown()
    }
  }, [releaseAudio, teardown])

  const toggle = useCallback(() => {
    const session = sessionRef.current
    if (!session) void start()
    else if (!session.stopped) stop()
    // A flush is pending, so ignore the click.
  }, [start, stop])

  // Never leave the mic running if the component goes away mid-session.
  useEffect(() => teardown, [teardown])

  return { status, interim, error, start, stop, toggle }
}
