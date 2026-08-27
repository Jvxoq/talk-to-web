/**
 * Forwards the API routes to the backend, same-origin.
 *
 * The browser only ever talks to this Vercel deployment. Everything under the
 * matched paths is passed on to `BACKEND_URL`, and the answer is streamed back
 * unchanged. That is what keeps the refresh cookie first-party: the browser
 * sees one origin, so it stores the cookie without SameSite=None.
 *
 * This used to be a list of `rewrites` in `vercel.json` with the backend host
 * written into every entry. Vercel reads that file before the build, so it
 * cannot take the host from an environment variable. Middleware runs per
 * request, so it can - set `BACKEND_URL` in the Vercel project settings.
 *
 * The WebSocket is deliberately not here: an Upgrade handshake does not survive
 * this hop either, which is why `useVoiceInput` dials the backend directly
 * through `VITE_WS_URL`.
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
