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
      // ws: true makes Vite forward the Upgrade handshake instead of
      // treating it as a plain HTTP request.
      '/ws': { target: backendUrl, ws: true },
    },
  },
})
