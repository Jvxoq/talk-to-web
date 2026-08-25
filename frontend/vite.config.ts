// `defineConfig` re-exported by vitest rather than vite: same function, but the
// type carries the `test` block below.
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
const backendUrl = process.env.VITE_BACKEND_URL ?? 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    proxy: {
      // Sign in, sign up, refresh and sign out. Proxied rather than called
      // cross-origin so the refresh cookie is first-party in development —
      // a `SameSite=None; Secure` cookie is never stored over plain http.
      '/auth': backendUrl,
      '/generate': backendUrl,
      '/upload': backendUrl,
      // Removing an attachment. Same list as nginx.conf and vercel.json.
      '/documents': backendUrl,
      '/conversations': backendUrl,
      // Without this the model list request is served by Vite itself, which
      // answers 200 with index.html - so the fetch succeeds, the JSON parse
      // fails, and the composer quietly falls back to its hardcoded models
      // instead of showing anything went wrong.
      '/models': backendUrl,
      // ws: true makes Vite forward the Upgrade handshake instead of
      // treating it as a plain HTTP request.
      '/ws': { target: backendUrl, ws: true },
    },
  },
  // Both units under test are plain modules — no DOM, so no jsdom dependency.
  // A component test added later needs `environment: 'jsdom'` and the package.
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
})
