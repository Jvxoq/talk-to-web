import { useCallback, useEffect, useRef, useState } from 'react'

export type VoiceStatus = 'idle' | 'connecting' | 'listening'

const WS_PATH = '/ws/transcribe/'
// Deepgram's recommended rate for speech: enough fidelity, a third of the
// bandwidth of 48kHz. The browser resamples the mic stream to match.
const TARGET_SAMPLE_RATE = 16000

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
}

function socketUrl(): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}${WS_PATH}`
}

export const isVoiceInputSupported = () =>
  typeof window !== 'undefined' &&
  !!navigator.mediaDevices?.getUserMedia &&
  typeof window.AudioWorkletNode !== 'undefined'

/**
 * Streams microphone audio to the backend over a WebSocket and surfaces the
 * transcripts coming back.
 *
 * Deepgram returns two kinds of result. Interim results are guesses that get
 * revised, so they live in `interim` and are meant to be rendered as a preview.
 * Final results are committed text and are handed to `onFinalTranscript` to be
 * appended to the composer.
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
    if (!session) return
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

      // Asking for 16kHz here makes the browser resample for us, so the
      // worklet only has to handle the float32 -> int16 conversion.
      context = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE })
      await context.audioWorklet.addModule('/pcm-worklet.js')
      node = new AudioWorkletNode(context, 'pcm-encoder')

      // An AudioWorkletNode is only pulled if it reaches the destination, so
      // route it through a muted gain node - we want the samples, not playback.
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

    const socket = new WebSocket(socketUrl())
    socket.binaryType = 'arraybuffer'
    sessionRef.current = { socket, context, stream, node }

    socket.onopen = () => {
      // Report the rate the AudioContext actually settled on - browsers do not
      // always honour the requested one, and a mismatch silently garbles pitch.
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
    if (!session) return

    // Stop capturing immediately so the mic indicator goes away, but leave the
    // socket open: the server still owes us the flushed final transcript, and
    // will reply "done" once Deepgram has emitted it.
    releaseAudio()
    if (session.socket.readyState === WebSocket.OPEN) {
      session.socket.send(JSON.stringify({ type: 'stop' }))
    } else {
      teardown()
    }
  }, [releaseAudio, teardown])

  const toggle = useCallback(() => {
    if (sessionRef.current) stop()
    else void start()
  }, [start, stop])

  // Never leave the mic running if the component goes away mid-session.
  useEffect(() => teardown, [teardown])

  return { status, interim, error, start, stop, toggle }
}
