import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
const backendUrl = process.env.VITE_BACKEND_URL ?? 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    proxy: {
      '/generate': backendUrl,
      '/upload': backendUrl,
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
})
