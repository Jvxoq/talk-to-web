# Talk to the Web

A full-stack AI application that lets you chat with web pages and PDF documents using Retrieval-Augmented Generation (RAG).

Built with **FastAPI** + **React** + **Qdrant** to demonstrate production-grade architectural patterns for building generative AI services.

---

## Implemented Architectural Concepts

This project serves as a reference implementation for the following patterns and concepts. Use it as a learning resource or starter template.

### Backend (FastAPI)

| Concept | Implementation |
|---|---|
| **Dependency Injection** | FastAPI's `Depends()` for injecting URL content, RAG content, and services into handlers |
| **Pydantic Schemas** | Request/response validation with `Literal` types for model selection |
| **Service Layer Pattern** | `VectorService` extends `VectorRepository` to separate business logic from data access |
| **Repository Pattern** | `VectorRepository` encapsulates all Qdrant vector database operations |
| **Lifespan Management** | `@asynccontextmanager` for startup/shutdown initialization of LLM client |
| **Background Tasks** | `BackgroundTasks` for non-blocking PDF processing and vector storage |
| **Streaming Responses** | `StreamingResponse` with async generators for real-time LLM output |
| **Async I/O** | `AsyncGroq`, `aiohttp`, `aiofiles`, and `asyncio.gather()` for non-blocking operations |
| **File Upload Handling** | Multipart uploads with content-type validation and chunked reading |
| **Vector Database (RAG)** | Qdrant integration with semantic search, cosine similarity, and collection management |
| **Text Processing Pipeline** | URL extraction, web scraping, PDF extraction, chunking, and embedding generation |
| **Error Handling** | Structured `HTTPException` with appropriate status codes and logging |
| **Configuration Management** | `.env` files with `python-dotenv` and centralized `config.py` |
| **Logging** | `loguru` for structured debug/warning/error logging |

### Frontend (React + TypeScript)

| Concept | Implementation |
|---|---|
| **Server-Sent Events (SSE)** | `ReadableStream` for real-time streaming from the backend |
| **State Management** | React hooks (`useState`, `useRef`) for messages, uploads, and UI state |
| **Environment Variables** | `import.meta.env.VITE_*` with fallback values via Vite |
| **CSS Custom Properties** | Design tokens for light/dark theme switching via `data-theme` |
| **Responsive Design** | Mobile-first CSS with flexbox layouts |
| **File Upload UI** | FormData construction with progress states and error handling |
| **TypeScript** | Interface definitions for messages, uploads, and model types |

### Cross-Cutting

| Concept | Implementation |
|---|---|
| **Separation of Concerns** | Clear module boundaries: routes, services, repositories, schemas, utils |
| **API Design** | RESTful endpoints with consistent response schemas |
| **Type Safety** | Type hints throughout backend, TypeScript on frontend |
| **CORS Configuration** | Specific origins, methods, and headers |

---

## Tech Stack

**Backend:** Python 3.13, FastAPI, Groq (LLM), Google Gemini (embeddings), Qdrant (vectors), aiohttp, pypdf

**Frontend:** React 19, TypeScript, Vite 8, Oxlint

---

## Getting Started

### Prerequisites

- Python 3.13+
- Node.js 18+
- Qdrant instance (local or cloud)
- API keys: [Groq](https://console.groq.com), [Google AI Studio](https://aistudio.google.com)

### Backend Setup

```bash
cd backend
cp .env.example .env   # Fill in your API keys
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn main:app --reload
```

### Frontend Setup

```bash
cd frontend
cp .env.example .env   # Optional: configure API URL
npm install
npm run dev
```

---

## Project Structure

```
talk-to-the-web/
├── backend/
│   ├── main.py              # App entry, routes, lifespan
│   ├── config.py            # Environment variable loading
│   ├── dependencies.py      # Business logic, DI functions
│   ├── schemas.py           # Pydantic request/response models
│   ├── utils.py             # LLM streaming utilities
│   ├── scraper.py           # Web scraping with BeautifulSoup
│   └── rag/
│       ├── extractor.py     # PDF text extraction
│       ├── transformer.py   # Embedding generation (Gemini)
│       ├── repository.py    # Qdrant vector operations
│       └── service.py       # RAG orchestration logic
└── frontend/
    └── src/
        └── App.tsx          # Chat UI with streaming support
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/generate/text/` | Generate text with optional URL context and PDF RAG |
| `POST` | `/upload/file/` | Upload a PDF for RAG indexing |

---

## License

MIT
