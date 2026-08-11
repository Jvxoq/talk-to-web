/// <reference types="vite/client" />

/**
 * The environment variables this app reads.
 *
 * `vite/client` types `import.meta.env` with an `any` index signature, so every
 * read is untyped by default. Declaring the ones we use merges into that
 * interface and gives them a real type — `string | undefined` rather than `any`
 * — which is what makes the `??` defaults at each read site meaningful, and what
 * documents the full set in one place.
 *
 * All are optional: each read site falls back to a same-origin path, which is
 * why `npm run dev` works with no `.env` file at all.
 */
interface ImportMetaEnv {
  /**
   * Auth base path. Defaults to `/auth`.
   *
   * Best left same-origin. The refresh token lives in a cookie the browser
   * attributes to whichever host answered, so proxying `/auth` through the
   * static host (Vercel rewrite, nginx `location`, or Vite's dev proxy) keeps
   * it first-party — and a first-party cookie needs neither `SameSite=None` nor
   * https to be stored.
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
   * Absolute `wss://` URL for the transcription socket. Set in production, where
   * the frontend is on Vercel and a rewrite cannot carry a WebSocket upgrade.
   */
  readonly VITE_WS_URL?: string
}
