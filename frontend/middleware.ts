/**
 * Forwards the API routes to the backend, same-origin.
 *
 * Everything under the matched paths goes to `BACKEND_URL` and the answer is
 * streamed back unchanged. The browser sees one origin, which is what keeps the
 * refresh cookie first-party.
 *
 * This was a list of `rewrites` in `vercel.json` with the backend host in every
 * entry. Vercel parses that file before the build, so it has no environment
 * variables. Middleware runs per request and does.
 *
 * The WebSocket is not here: an Upgrade handshake does not survive this hop
 * either, so `useVoiceInput` connects to the backend directly via `VITE_WS_URL`.
 */

export const config = {
  matcher: [
    '/auth/:path*',
    '/generate/:path*',
    '/upload/:path*',
    '/documents/:path*',
    '/conversations/:path*',
    '/models/:path*',
    '/health',
  ],
}

export default async function middleware(request: Request): Promise<Response> {
  const backend = process.env.BACKEND_URL

  if (!backend) {
    return new Response(
      JSON.stringify({ detail: 'BACKEND_URL is not set on this deployment' }),
      { status: 500, headers: { 'content-type': 'application/json' } },
    )
  }

  const incoming = new URL(request.url)
  const target = new URL(incoming.pathname + incoming.search, backend)

  // `redirect: 'manual'` so a 3xx from the backend reaches the browser as a
  // 3xx, instead of being followed here and answered from the wrong URL.
  return fetch(new Request(target, request), { redirect: 'manual' })
}
