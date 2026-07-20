// Runs on the audio rendering thread, off the main thread.
//
// The browser hands us float32 samples in [-1, 1], 128 frames at a time.
// Deepgram wants linear16: signed 16-bit little-endian PCM. We convert, and
// batch up to BATCH_SAMPLES before posting so we send ~64ms chunks instead of
// a 128-sample (8ms) WebSocket frame 125 times a second.
const BATCH_SAMPLES = 1024

class PCMEncoder extends AudioWorkletProcessor {
  constructor() {
    super()
    this.buffer = new Int16Array(BATCH_SAMPLES)
    this.offset = 0
  }

  process(inputs) {
    const channel = inputs[0]?.[0]
    // No input connected yet (or the track ended); keep the node alive.
    if (!channel) return true

    for (let i = 0; i < channel.length; i++) {
      // Clamp before scaling: values can drift slightly outside [-1, 1].
      const sample = Math.max(-1, Math.min(1, channel[i]))
      // Asymmetric scaling because int16 range is [-32768, 32767].
      this.buffer[this.offset++] = sample < 0 ? sample * 0x8000 : sample * 0x7fff

      if (this.offset === BATCH_SAMPLES) {
        // Transfer the buffer rather than copying it across the thread boundary.
        const chunk = this.buffer.buffer
        this.port.postMessage(chunk, [chunk])
        this.buffer = new Int16Array(BATCH_SAMPLES)
        this.offset = 0
      }
    }

    return true
  }
}

registerProcessor('pcm-encoder', PCMEncoder)
