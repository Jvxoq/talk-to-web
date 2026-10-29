// From vitest, so the type includes the `test` block.
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
const backendUrl = process.env.VITE_BACKEND_URL ?? 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    proxy: {
      // Proxied, so the refresh cookie is first-party over plain http.
      '/auth': backendUrl,
      '/generate': backendUrl,
      '/upload': backendUrl,
      // Removing an attachment. Keep in step with `middleware.ts`.
      '/documents': backendUrl,
      '/conversations': backendUrl,
      // Without this, Vite answers with index.html and the model list breaks quietly.
      '/models': backendUrl,
      // `BackendGate`'s probe. Without it, Vite answers with index.html.
      '/health': backendUrl,
      // Forward the WebSocket upgrade.
      '/ws': { target: backendUrl, ws: true },
    },
  },
  // No DOM. A component test needs jsdom installed and `environment: 'jsdom'`.
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
})
