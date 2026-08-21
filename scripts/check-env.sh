#!/usr/bin/env bash
# Fails loudly if backend/.env.production still has a placeholder value in it.
# Run before `docker compose -f docker-compose.prod.yml up` — nothing here
# validates that credentials are *correct*, only that they were changed from
# the checked-in example at all.
#
#   ./scripts/check-env.sh

set -euo pipefail

env_file="backend/.env.production"

if [[ ! -f "$env_file" ]]; then
  echo "check-env: $env_file does not exist — cp backend/.env.production.example $env_file first" >&2
  exit 1
fi

placeholders=(
  "postgres:postgres@postgres:5432"
  "generate_a_long_random_secret_for_this_deployment"
  "your_qdrant_cloud_api_key"
  "your_llm_api_key_here"
  "your_tavily_api_key_here"
  "your_gemini_api_key_here"
  "your_deepgram_api_key_here"
  "your-app.vercel.app"
  "ep-example-123456"
)

found=0
for placeholder in "${placeholders[@]}"; do
  if grep -q -- "$placeholder" "$env_file"; then
    echo "check-env: $env_file still has placeholder value: $placeholder" >&2
    found=1
  fi
done

if [[ "$found" -eq 1 ]]; then
  echo "check-env: fix the lines above before deploying" >&2
  exit 1
fi

echo "check-env: $env_file has no known placeholders"
