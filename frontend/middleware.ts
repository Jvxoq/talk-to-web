/**
 * Forwards the API routes to `BACKEND_URL`, so the browser sees one origin.
 *
 * Middleware, not `vercel.json`, because only middleware can read env vars.
 * WebSockets cannot pass through here, so `useVoiceInput` uses `VITE_WS_URL`.
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

  // Pass a backend 3xx through to the browser instead of following it.
  return fetch(new Request(target, request), { redirect: 'manual' })
}
