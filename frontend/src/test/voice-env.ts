/**
 * Fakes for `getUserMedia`, Web Audio and `WebSocket`, which jsdom lacks.
 *
 * Only the platform is faked. The hook's own logic stays real.
 */

import { vi } from 'vitest'

/** `WebSocket.readyState` values the hook compares against. */
export const CONNECTING = 0
export const OPEN = 1
export const CLOSING = 2
export const CLOSED = 3

export interface FakeSocket {
  url: string
  protocols: string[]
  readyState: number
  sent: unknown[]
  closeCount: number
  binaryType: string
  onopen: (() => void) | null
  onmessage: ((event: { data: string }) => void) | null
  onerror: (() => void) | null
  onclose: (() => void) | null
  send(data: unknown): void
  close(): void
  /** Drives the socket open, as a server accept would. */
  open(): void
  /** Feeds one JSON frame to the app. */
  receive(message: Record<string, unknown>): void
  /** The connection fails without the app asking. */
  fail(): void
  /** The peer closes first. */
  closeFromPeer(): void
}

export interface FakeTrack {
  stop: ReturnType<typeof vi.fn>
}

export interface FakeStream {
  getTracks(): FakeTrack[]
}

export interface FakeGraphNode {
  gain: { value: number }
  connect(target: unknown): unknown
}

export interface FakeAudioContext {
  sampleRate: number
  destination: object
  close: ReturnType<typeof vi.fn>
  audioWorklet: { addModule: ReturnType<typeof vi.fn> }
  createGain(): FakeGraphNode
  createMediaStreamSource(stream: FakeStream): FakeGraphNode
}

export interface FakeWorkletNode {
  port: { onmessage: ((event: { data: ArrayBuffer }) => void) | null }
  /** Real `AudioNode.connect` returns the destination node, so calls chain. */
  connect(target: unknown): FakeGraphNode
  disconnect: ReturnType<typeof vi.fn>
}

/** Every WebSocket the app opened this test, in construction order. */
export const sockets: FakeSocket[] = []
/** Every AudioContext the app constructed this test. */
export const audioContexts: FakeAudioContext[] = []
/** Every `AudioWorkletNode` the app constructed this test. */
export const workletNodes: FakeWorkletNode[] = []

/** The one track `getUserMedia` hands out, so a test can assert it was stopped. */
export const track: FakeTrack = { stop: vi.fn() }
export const stream: FakeStream = { getTracks: () => [track] }

/**
 * The default `getUserMedia` fake. A `vi.fn`, so tests can make it reject.
 */
export const getUserMedia = vi.fn(async (): Promise<FakeStream> => stream)

function graphNode(): FakeGraphNode {
  return { gain: { value: 1 }, connect: (target) => target }
}

class FakeWebSocket {
  static readonly CONNECTING = CONNECTING
  static readonly OPEN = OPEN
  static readonly CLOSING = CLOSING
  static readonly CLOSED = CLOSED

  readonly url: string
  readonly protocols: string[]
  readonly sent: unknown[] = []
  readyState = CONNECTING
  closeCount = 0
  binaryType = 'blob'
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onerror: (() => void) | null = null
  onclose: (() => void) | null = null

  constructor(url: string, protocols: string[]) {
    this.url = url
    this.protocols = protocols
    sockets.push(this as unknown as FakeSocket)
  }

  send(data: unknown): void {
    if (this.readyState !== OPEN) throw new Error('FakeWebSocket: send() while not open')
    // Decode only the JSON control frames. Audio is raw binary.
    if (typeof data === 'string') {
      try {
        this.sent.push(JSON.parse(data))
        return
      } catch {
        // Not JSON - fall through and record it verbatim.
      }
    }
    this.sent.push(data)
  }

  close(): void {
    this.closeCount += 1
    this.readyState = CLOSED
  }

  open(): void {
    this.readyState = OPEN
    this.onopen?.()
  }

  receive(message: Record<string, unknown>): void {
    this.onmessage?.({ data: JSON.stringify(message) })
  }

  fail(): void {
    this.onerror?.()
  }

  closeFromPeer(): void {
    this.readyState = CLOSED
    this.onclose?.()
  }
}

/** The implementation behind the `FakeAudioContext` interface above. */
class FakeAudioContextImpl {
  readonly sampleRate: number
  readonly destination = {}
  readonly close = vi.fn(async () => {})
  readonly audioWorklet = { addModule: vi.fn(async () => {}) }

  constructor(options: { sampleRate?: number } = {}) {
    // Use the requested rate, so a test can choose any.
    this.sampleRate = options.sampleRate ?? 16000
    audioContexts.push(this as unknown as FakeAudioContext)
  }

  createGain(): FakeGraphNode {
    return graphNode()
  }

  createMediaStreamSource(): FakeGraphNode {
    return graphNode()
  }
}

class FakeAudioWorkletNode {
  readonly port: { onmessage: ((event: { data: ArrayBuffer }) => void) | null } = {
    onmessage: null,
  }
  readonly disconnect = vi.fn()

  constructor() {
    workletNodes.push(this as unknown as FakeWorkletNode)
  }

  connect(_target: unknown): FakeGraphNode {
    return graphNode()
  }
}

let mediaDevicesDescriptor: PropertyDescriptor | undefined

/**
 * Installs the three fakes globally.
 *
 * `supported: false` models a browser without Web Audio.
 */
export function installVoiceEnv({ supported = true }: { supported?: boolean } = {}) {
  sockets.length = 0
  audioContexts.length = 0
  workletNodes.length = 0
  track.stop.mockClear()
  getUserMedia.mockReset()
  getUserMedia.mockImplementation(async () => stream)

  if (!supported) {
    vi.stubGlobal('AudioWorkletNode', undefined)
    Object.defineProperty(navigator, 'mediaDevices', {
      value: undefined,
      configurable: true,
      writable: true,
    })
    return
  }

  vi.stubGlobal('WebSocket', FakeWebSocket)
  vi.stubGlobal('AudioContext', FakeAudioContextImpl)
  vi.stubGlobal('AudioWorkletNode', FakeAudioWorkletNode)

  mediaDevicesDescriptor = Object.getOwnPropertyDescriptor(navigator, 'mediaDevices')
  Object.defineProperty(navigator, 'mediaDevices', {
    value: { getUserMedia },
    configurable: true,
    writable: true,
  })
}

/** Puts the globals back. Call from `afterEach`. */
export function uninstallVoiceEnv() {
  vi.unstubAllGlobals()
  if (mediaDevicesDescriptor) {
    Object.defineProperty(navigator, 'mediaDevices', mediaDevicesDescriptor)
    mediaDevicesDescriptor = undefined
  } else {
    delete (navigator as { mediaDevices?: unknown }).mediaDevices
  }
}

/** The last socket the app opened, or a thrown error if it opened none. */
export function lastSocket(): FakeSocket {
  const socket = sockets.at(-1)
  if (!socket) throw new Error('no WebSocket was opened')
  return socket
}

/** The last worklet node the app built, or a thrown error if it built none. */
export function lastWorkletNode(): FakeWorkletNode {
  const node = workletNodes.at(-1)
  if (!node) throw new Error('no AudioWorkletNode was constructed')
  return node
}

/** The last audio context the app built, or a thrown error if it built none. */
export function lastAudioContext(): FakeAudioContext {
  const context = audioContexts.at(-1)
  if (!context) throw new Error('no AudioContext was constructed')
  return context
}