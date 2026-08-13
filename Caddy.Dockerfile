# Stock `caddy:2-alpine` has no rate-limiting module - the `rate_limit`
# directive in `Caddyfile` only exists once this plugin is compiled in. Built
# with xcaddy rather than pulled, because the plugin ships as source, not a
# prebuilt binary.
FROM caddy:2-builder-alpine AS builder

RUN xcaddy build --with github.com/mholt/caddy-ratelimit

FROM caddy:2-alpine

COPY --from=builder /usr/bin/caddy /usr/bin/caddy
