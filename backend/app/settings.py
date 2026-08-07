"""The only module that reads the environment."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = "local"
    log_level: str = "DEBUG"

    # --- HTTP ---
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    request_timeout_seconds: float = 30.0

    # --- Postgres ---
    database_url: str
    database_pool_size: int = 10
    database_max_overflow: int = 5

    # --- LLM (provider-agnostic) ---
    # The provider is a string because the adapter resolves it through
    # `init_chat_model`, which accepts "groq", "openai", "anthropic",
    # "google_genai", "ollama" and more. Switching provider is an env change.
    llm_provider: str = "groq"
    # Every entry must support tool calling, and support it *reliably* - the
    # agent is useless without it. Two models are gone for failing that bar,
    # each for its own reason:
    #
    #   groq/compound            refuses a custom `tools` payload outright; it
    #                            only runs Groq's own built-in tools.
    #   llama-3.3-70b-versatile  accepts tools but emits malformed calls. Measured
    #                            against these tool schemas it managed 1/3, and
    #                            0/3 on the search tool, failing the whole reply
    #                            with "Failed to call a function".
    #
    # The two below were measured at 3/3 on the same probe. The first is the
    # default the UI offers; the second is the smaller, faster option.
    llm_models: list[str] = Field(
        default_factory=lambda: ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    )
    llm_api_key: SecretStr
    llm_max_tokens: int = 2048
    llm_system_prompt: str = (
        "You are a helpful assistant with access to tools.\n"
        "Use `retrieve_documents` for questions about the user's uploaded files.\n"
        "Use `fetch_web_pages` when the user gives a URL they want read.\n"
        "Use `search_web` for current information you do not already know.\n"
        "Answer directly, without calling a tool, when you already know the answer."
    )

    # --- Agent ---
    # The ceiling on model -> tools -> model round trips in a single reply. A
    # model that keeps asking for tools is looping, and every lap costs a
    # request the user is waiting on.
    agent_max_tool_iterations: int = 5

    # --- Tavily (web search) ---
    tavily_api_key: SecretStr
    tavily_max_results: int = 5

    # --- Gemini (embeddings) ---
    gemini_api_key: SecretStr
    embedding_model: str = "gemini-embedding-2"
    embedding_dimensions: int = 768

    # --- Qdrant (vector store) ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "knowledge_base"
    retrieval_limit: int = 3
    retrieval_score_threshold: float = 0.7

    # --- Deepgram (speech to text) ---
    deepgram_api_key: SecretStr
    deepgram_model: str = "nova-3"
    utterance_end_ms: int = 1500
    finalize_timeout_seconds: float = 3.0

    # --- Ingestion ---
    upload_dir: Path = Path("uploads")
    static_pages_dir: Path = Path("pages")
    max_upload_bytes: int = 25 * 1024 * 1024
    chunk_size: int = 500
    chunk_overlap: int = 50


@lru_cache
def get_settings() -> Settings:
    """Read the environment once. Called from `create_app` and nowhere else."""
    return Settings()
