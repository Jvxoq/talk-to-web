# Talk to the Web

A full-stack AI chat application that lets you upload PDFs or paste URLs, then chat about their content using Retrieval-Augmented Generation (RAG). Built with FastAPI, React, Qdrant, Groq (LLM), and Deepgram (live speech-to-text).

## Getting Started

```bash
# Backend
cd backend
cp .env.example .env
uv sync --all-groups
uv run fastapi dev app/main.py

# Frontend (in another terminal)
cd frontend
npm install
npm run dev
```

The app runs on `localhost:5173` and connects to the backend at `localhost:8000`.
