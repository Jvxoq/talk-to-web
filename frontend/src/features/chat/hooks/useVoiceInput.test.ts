// @vitest-environment jsdom
// Tests for useVoiceInput: the start/stop/teardown state machine, the
// WebSocket handshake, and how transcripts and failures surface.

import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { clearAccessToken, setAccessToken } from '../../../lib/session'
import {
  installVoiceEnv,
  lastAudioContext,
  lastSocket,
  lastWorkletNode,
  track,
  uninstallVoiceEnv,
} from '../../../test/voice-env'
import { isVoiceInputSupported, useVoiceInput } from './useVoiceInput'

const TOKEN = 'test-access-token'

/** Starts a session and returns once the socket exists but is not yet open. */
function startSession(onFinal: (text: string) => void = vi.fn()) {
  const view = renderHook(() => useVoiceInput(onFinal))
  return { view, onFinal }
}

beforeEach(() => {
  installVoiceEnv()
  setAccessToken(TOKEN)
})

afterEach(() => {
  clearAccessToken()
  uninstallVoiceEnv()
})

describe('isVoiceInputSupported', () => {
  it('is true where getUserMedia and AudioWorkletNode both exist', () => {
    expect(isVoiceInputSupported()).toBe(true)
  })

  it('is false in a browser without Web Audio, so the button can disable itself', () => {
    installVoiceEnv({ supported: false })

    expect(isVoiceInputSupported()).toBe(false)
  })
})

describe('useVoiceInput', () => {
  it('connects, then listens once the server says ready', async () => {
    const { view } = startSession()

    expect(view.result.current.status).toBe('idle')

    await act(async () => {
      await view.result.current.start()
    })
    expect(view.result.current.status).toBe('connecting')

    await act(async () => {
      lastSocket().open()
    })
    expect(view.result.current.status).toBe('connecting')

    await act(async () => {
      lastSocket().receive({ type: 'ready' })
    })
    expect(view.result.current.status).toBe('listening')
  })

  it('reports the rate the audio context settled on, since browsers pick their own', async () => {
    const { view } = startSession()
    await act(async () => {
      await view.result.current.start()
    })

    await act(async () => {
      lastSocket().open()
    })

    expect(lastSocket().sent).toEqual([{ type: 'start', sample_rate: 16000 }])
  })

  it('sends the token as a subprotocol and never in the URL', async () => {
    const { view } = startSession()
    await act(async () => {
      await view.result.current.start()
    })

    expect(lastSocket().protocols).toEqual(['access_token', TOKEN])
    expect(lastSocket().url).toContain('/ws/transcribe/')
    expect(lastSocket().url).not.toContain(TOKEN)
  })

  it('keeps a preview of interim text without committing it', async () => {
    const { view, onFinal } = startSession()
    await act(async () => {
      await view.result.current.start()
    })
    await act(async () => {
      lastSocket().open()
    })

    await act(async () => {
      lastSocket().receive({ type: 'transcript', text: 'hello ther', is_final: false })
    })

    expect(view.result.current.interim).toBe('hello ther')
    expect(onFinal).not.toHaveBeenCalled()
  })

  it('commits a final transcript and drops the preview', async () => {
    const { view, onFinal } = startSession()
    await act(async () => {
      await view.result.current.start()
    })
    await act(async () => {
      lastSocket().open()
    })
    await act(async () => {
      lastSocket().receive({ type: 'transcript', text: 'hello there', is_final: false })
    })

    await act(async () => {
      lastSocket().receive({ type: 'transcript', text: 'hello there.', is_final: true })
    })

    expect(onFinal).toHaveBeenCalledWith('hello there.')
    expect(view.result.current.interim).toBe('')
  })

  it('commits nothing for a final result carrying no text', async () => {
    // An empty final (silence) must not add a space.
    const { view, onFinal } = startSession()
    await act(async () => {
      await view.result.current.start()
    })
    await act(async () => {
      lastSocket().open()
    })
    await act(async () => {
      lastSocket().receive({ type: 'transcript', text: 'partial', is_final: false })
    })

    await act(async () => {
      lastSocket().receive({ type: 'transcript', is_final: true })
    })

    expect(onFinal).not.toHaveBeenCalled()
    expect(view.result.current.interim).toBe('')
  })

  it('releases the mic and the socket when the server says done', async () => {
    const { view } = startSession()
    await act(async () => {
      await view.result.current.start()
    })
    await act(async () => {
      lastSocket().open()
    })
    await act(async () => {
      lastSocket().receive({ type: 'ready' })
    })

    await act(async () => {
      lastSocket().receive({ type: 'done' })
    })

    expect(view.result.current.status).toBe('idle')
    expect(view.result.current.interim).toBe('')
    expect(track.stop).toHaveBeenCalledTimes(1)
    expect(lastWorkletNode().disconnect).toHaveBeenCalledTimes(1)
    expect(lastAudioContext().close).toHaveBeenCalledTimes(1)
    expect(lastSocket().closeCount).toBe(1)
  })

  it('surfaces the server detail on an error frame, then tears down', async () => {
    const { view } = startSession()
    await act(async () => {
      await view.result.current.start()
    })
    await act(async () => {
      lastSocket().open()
    })

    await act(async () => {
      lastSocket().receive({ type: 'error', detail: 'Deepgram rejected the stream' })
    })

    expect(view.result.current.error).toBe('Deepgram rejected the stream')
    expect(view.result.current.status).toBe('idle')
    expect(track.stop).toHaveBeenCalledTimes(1)
    expect(lastSocket().closeCount).toBe(1)
  })

  it('has a message for an error frame carrying no detail', async () => {
    const { view } = startSession()
    await act(async () => {
      await view.result.current.start()
    })
    await act(async () => {
      lastSocket().open()
    })

    await act(async () => {
      lastSocket().receive({ type: 'error' })
    })

    expect(view.result.current.error).toBe('Transcription failed.')
  })

  it('reports a dropped connection and tears down', async () => {
    const { view } = startSession()
    await act(async () => {
      await view.result.current.start()
    })
    await act(async () => {
      lastSocket().open()
    })
    await act(async () => {
      lastSocket().receive({ type: 'ready' })
    })

    await act(async () => {
      lastSocket().fail()
    })

    expect(view.result.current.error).toBe('Lost connection to the transcription service.')
    expect(view.result.current.status).toBe('idle')
    expect(track.stop).toHaveBeenCalledTimes(1)
  })

  it('tears down when the peer closes the socket first', async () => {
    const { view } = startSession()
    await act(async () => {
      await view.result.current.start()
    })
    await act(async () => {
      lastSocket().open()
    })

    await act(async () => {
      lastSocket().closeFromPeer()
    })

    expect(view.result.current.status).toBe('idle')
    expect(track.stop).toHaveBeenCalledTimes(1)
  })

  it('reports a denied microphone and asks for no socket', async () => {
    const { getUserMedia } = await import('../../../test/voice-env')
    getUserMedia.mockRejectedValueOnce(
      new DOMException('Permission denied', 'NotAllowedError'),
    )
    const { view } = startSession()

    await act(async () => {
      await view.result.current.start()
    })

    expect(view.result.current.error).toBe('Microphone access was denied.')
    expect(view.result.current.status).toBe('idle')
  })

  it('reports a microphone that failed for any other reason', async () => {
    const { getUserMedia } = await import('../../../test/voice-env')
    getUserMedia.mockRejectedValueOnce(new Error('audio hardware busy'))
    const { view } = startSession()

    await act(async () => {
      await view.result.current.start()
    })

    expect(view.result.current.error).toBe('Could not start the microphone.')
    expect(view.result.current.status).toBe('idle')
  })

  it('recovers on the next attempt after a failed one', async () => {
    const { getUserMedia } = await import('../../../test/voice-env')
    getUserMedia.mockRejectedValueOnce(new Error('audio hardware busy'))
    const { view } = startSession()
    await act(async () => {
      await view.result.current.start()
    })

    await act(async () => {
      await view.result.current.start()
    })

    expect(view.result.current.error).toBeNull()
    expect(view.result.current.status).toBe('connecting')
  })

  it('reports an expired session without ever touching the microphone', async () => {
    // No token, so the mic is never requested.
    const { getUserMedia } = await import('../../../test/voice-env')
    clearAccessToken()
    const { view } = startSession()

    await act(async () => {
      await view.result.current.start()
    })

    expect(view.result.current.error).toBe('Your session has expired. Please sign in again.')
    expect(view.result.current.status).toBe('idle')
    expect(getUserMedia).not.toHaveBeenCalled()
  })

  it('refuses a second start while a session is live', async () => {
    const { view } = startSession()
    await act(async () => {
      await view.result.current.start()
    })
    const opened = lastSocket()

    await act(async () => {
      await view.result.current.start()
    })

    expect(opened.closeCount).toBe(0)
    expect(lastSocket()).toBe(opened)
  })

  it('frees the mic on stop but leaves the socket open for the final flush', async () => {
    const { view } = startSession()
    await act(async () => {
      await view.result.current.start()
    })
    await act(async () => {
      lastSocket().open()
    })
    await act(async () => {
      lastSocket().receive({ type: 'ready' })
    })

    await act(async () => {
      view.result.current.stop()
    })

    expect(view.result.current.status).toBe('stopping')
    expect(track.stop).toHaveBeenCalledTimes(1)
    expect(lastWorkletNode().disconnect).toHaveBeenCalledTimes(1)
    expect(lastAudioContext().close).toHaveBeenCalledTimes(1)
    expect(lastSocket().sent).toContainEqual({ type: 'stop' })
    expect(lastSocket().closeCount).toBe(0)
  })

  it('ends the session when stop lands before the socket ever opened', async () => {
    const { view } = startSession()
    await act(async () => {
      await view.result.current.start()
    })

    await act(async () => {
      view.result.current.stop()
    })

    expect(view.result.current.status).toBe('idle')
    expect(track.stop).toHaveBeenCalledTimes(1)
    expect(lastSocket().closeCount).toBe(1)
  })

  it('sends one stop frame however many times stop is called', async () => {
    const { view } = startSession()
    await act(async () => {
      await view.result.current.start()
    })
    await act(async () => {
      lastSocket().open()
    })

    await act(async () => {
      view.result.current.stop()
    })
    await act(async () => {
      view.result.current.stop()
    })

    expect(lastSocket().sent.filter((m) => JSON.stringify(m) === '{"type":"stop"}')).toHaveLength(1)
    expect(track.stop).toHaveBeenCalledTimes(1)
  })

  it('returns to idle and lets the next session start once the flush lands', async () => {
    const { view } = startSession()
    await act(async () => {
      await view.result.current.start()
    })
    await act(async () => {
      lastSocket().open()
    })
    await act(async () => {
      view.result.current.stop()
    })

    await act(async () => {
      lastSocket().receive({ type: 'transcript', text: 'the tail', is_final: true })
      lastSocket().receive({ type: 'done' })
    })

    expect(view.result.current.status).toBe('idle')

    await act(async () => {
      await view.result.current.start()
    })
    expect(view.result.current.status).toBe('connecting')
  })

  it('forwards worklet audio to the socket only while it is open', async () => {
    const { view } = startSession()
    await act(async () => {
      await view.result.current.start()
    })

    // No handler before the handshake, so frames are dropped.
    expect(lastWorkletNode().port.onmessage).toBeNull()

    await act(async () => {
      lastSocket().open()
    })
    const buffer = new ArrayBuffer(8)

    act(() => {
      lastWorkletNode().port.onmessage?.({ data: buffer })
    })

    expect(lastSocket().sent).toEqual([{ type: 'start', sample_rate: 16000 }, buffer])
  })

  it('never leaves the mic running when the component goes away', async () => {
    const { view } = startSession()
    await act(async () => {
      await view.result.current.start()
    })
    await act(async () => {
      lastSocket().open()
    })
    await act(async () => {
      lastSocket().receive({ type: 'ready' })
    })

    await act(async () => {
      view.unmount()
    })

    expect(track.stop).toHaveBeenCalledTimes(1)
    expect(lastWorkletNode().disconnect).toHaveBeenCalledTimes(1)
    expect(lastAudioContext().close).toHaveBeenCalledTimes(1)
    expect(lastSocket().closeCount).toBe(1)
  })

  describe('toggle', () => {
    it('starts from idle and stops from listening', async () => {
      const { view } = startSession()

      await act(async () => {
        view.result.current.toggle()
      })
      await waitFor(() => expect(lastSocket().readyState).toBe(0))
      await act(async () => {
        lastSocket().open()
      })
      await act(async () => {
        lastSocket().receive({ type: 'ready' })
      })

      await act(async () => {
        view.result.current.toggle()
      })

      expect(view.result.current.status).toBe('stopping')
      expect(lastSocket().sent).toContainEqual({ type: 'stop' })
    })

    it('ignores the click while a flush is pending, so the frame is not sent twice', async () => {
      const { view } = startSession()
      await act(async () => {
        await view.result.current.start()
      })
      await act(async () => {
        lastSocket().open()
      })
      await act(async () => {
        view.result.current.stop()
      })

      await act(async () => {
        view.result.current.toggle()
      })

      expect(view.result.current.status).toBe('stopping')
      expect(
        lastSocket().sent.filter((m) => JSON.stringify(m) === '{"type":"stop"}'),
      ).toHaveLength(1)
    })

    it('clears a previous failure before retrying', async () => {
      const { getUserMedia } = await import('../../../test/voice-env')
      getUserMedia.mockRejectedValueOnce(new Error('audio hardware busy'))
      const { view } = startSession()
      await act(async () => {
        await view.result.current.start()
      })
      expect(view.result.current.error).toBe('Could not start the microphone.')

      await act(async () => {
        await view.result.current.start()
      })

      expect(view.result.current.error).toBeNull()
    })
  })
})