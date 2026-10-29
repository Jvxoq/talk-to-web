/// <reference types="vite/client" />

/**
 * The environment variables this app reads, typed `string | undefined`.
 *
 * All optional: each falls back to same-origin, so `npm run dev` needs no `.env`.
 */
interface ImportMetaEnv {
  /**
   * Auth base path. Defaults to `/auth`.
   *
   * Keep it same-origin, so the refresh cookie stays first-party.
   */
  readonly VITE_AUTH_URL?: string
  /** Chat streaming endpoint. Defaults to `/generate/text/`. */
  readonly VITE_API_URL?: string
  /** PDF upload endpoint. Defaults to `/upload/file/`. */
  readonly VITE_UPLOAD_URL?: string
  /** Model list endpoint. Defaults to `/models/`. */
  readonly VITE_MODELS_URL?: string
  /** Conversations base path. Defaults to `/conversations/`. */
  readonly VITE_CONVERSATIONS_URL?: string
  /**
   * Absolute `wss://` URL for transcription. Needed on Vercel, which cannot proxy WebSockets.
   */
  readonly VITE_WS_URL?: string
}
