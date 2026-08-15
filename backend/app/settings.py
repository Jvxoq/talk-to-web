"""The only module that reads the environment."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = "local"
    log_level: str = "DEBUG"

    # --- Error reporting ---
    # Unset means Sentry is never initialized: no key to hold locally, nothing
    # sent from a test run, and one variable to set when a deployment wants it.
    sentry_dsn: SecretStr | None = None
    # What identifies the running build in Sentry - the image tag or commit SHA,
    # supplied by whatever does the deploy. Unset, Sentry groups every release
    # together and "regression since" stops meaning anything.
    sentry_release: str | None = None
    # Performance tracing, off by default. It samples whole requests, and this
    # app's requests are long: a chat reply holds a stream open for the length of
    # a model call. Turn it up deliberately, in small fractions.
    sentry_traces_sample_rate: float = 0.0

    # --- Tracing (Langfuse) ---
    # Unset means the tracer is never built: the app runs on `NullTracer`, no
    # key is held locally, and a test run sends nothing anywhere. Same posture
    # as `sentry_dsn` above, for the same reason.
    #
    # Cloud rather than self-hosted, deliberately. Self-hosting Langfuse is five
    # more containers - web, worker, ClickHouse, MinIO, Redis - plus its own
    # Postgres, and this deployment is a 1 GiB instance where the backend alone
    # is capped at 700m. The hosted free tier costs two variables.
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    # Whether prompts and completions ride along with the trace. On, because a
    # trace without the text is a latency chart. Safe *because* the input
    # guardrail redacts before any span is opened - turn it off for a deployment
    # where that is not enough.
    langfuse_capture_content: bool = True
    # How long shutdown waits for queued spans. Bounded, and far below the
    # compose `stop_grace_period` of 30s: a tracing host that has gone away must
    # not turn a deploy into a hang.
    langfuse_flush_timeout_seconds: float = 2.0

    # --- Guardrails ---
    # Redact secrets and personal data out of the user's message before it
    # reaches the model, the checkpointer or the trace. Everything downstream
    # sees the redacted text, which is what keeps a pasted API key from being
    # archived by three systems at once.
    guardrail_pii_redaction_enabled: bool = True
    # Whether a suspected prompt injection is refused or merely recorded.
    #
    # Off, deliberately, and not as timidity. "Ignore the previous paragraph and
    # summarize the rest" is a sentence a real user types at a real PDF, and a
    # heuristic that blocks it on day one refuses honest work to stop an attack
    # nobody has measured yet. Flag first, read the false-positive rate off
    # `evals/datasets/benign.jsonl`, then turn it on.
    guardrail_block_on_injection: bool = False
    # Strip instruction-shaped lines out of tool results before the model reads
    # them. The fence around untrusted content is always applied; this is the
    # second, sharper pass on top of it.
    guardrail_strip_tool_instructions: bool = True
    # How much text any single detector will scan.
    #
    # This is a concurrency limit wearing a size limit's clothes. Regex is
    # synchronous, this deployment runs one uvicorn worker with
    # `--limit-concurrency 5`, and Python offers no way to time a match out - so
    # a scan that runs long does not slow one reply, it stalls every open stream
    # on the box. A user message is capped at 32k by the request schema, but a
    # tool result is ten fetched pages at 20k each, and that is the input this
    # bound exists for.
    guardrail_max_scan_chars: int = 50_000

    # --- HTTP ---
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    # Origins allowed to open the speech-to-text WebSocket. CORS middleware does
    # not run on a WebSocket handshake — there is no preflight and no response to
    # attach headers to — so the browser's same-origin protection simply does not
    # apply here. Without this list, anyone who knows the URL can open a socket
    # against a public deployment and spend its Deepgram budget. Left empty it
    # mirrors `cors_origins`, which is the right answer in every topology we
    # deploy: the page allowed to call the API is the page allowed to talk.
    allowed_websocket_origins: list[str] = Field(default_factory=list)
    request_timeout_seconds: float = 30.0
    # How long one `/ready` probe may take before it counts as a failure. Short
    # on purpose, and shorter than any orchestrator's probe timeout: the point of
    # the endpoint is to answer quickly that this process cannot serve traffic,
    # and a probe that outlives the caller's patience reports nothing at all.
    readiness_timeout_seconds: float = Field(default=2.0, gt=0)

    # --- Postgres ---
    database_url: str
    # The URL `alembic upgrade head` connects with, when it must differ from the
    # one the application uses. Neon publishes two endpoints for one database: a
    # pooled one (PgBouncer, transaction mode) that the app should use, and a
    # direct one. DDL through a transaction pooler misbehaves — session state
    # like advisory locks and `SET LOCAL` does not survive a connection being
    # handed to another client mid-migration — so migrations take the direct
    # endpoint. Unset (the local case, where there is no pooler) it falls back to
    # `database_url`.
    database_migration_url: str | None = None
    database_pool_size: int = 10
    database_max_overflow: int = 5
    # Separate from the pool above: the checkpointer issues short autocommit
    # statements, not held transactions, so it needs far fewer connections than
    # the SQLAlchemy pool serving requests.
    checkpointer_pool_min_size: int = 1
    checkpointer_pool_max_size: int = 5

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
        "Use `retrieve_documents` for questions about the user's uploaded files, "
        "and also whenever the question names a specific person, company, product "
        "or other entity you do not already know - the user may have uploaded a "
        "document about it without saying so. Try this before `search_web` in "
        "that case; it is a cheap, private lookup and coming back empty costs "
        "only one extra call.\n"
        "Use `fetch_web_pages` when the user gives a URL they want read.\n"
        "Use `search_web` for current information you do not already know, or "
        "after `retrieve_documents` comes back empty for a named entity.\n"
        "Answer directly, without calling a tool, when you already know the answer."
    )

    # --- Agent ---
    # The ceiling on model -> tools -> model round trips in a single reply. A
    # model that keeps asking for tools is looping, and every lap costs a
    # request the user is waiting on.
    agent_max_tool_iterations: int = 5

    # --- Agent compression ---
    # The agent replays the whole thread on every lap (agent -> tools -> agent),
    # so a long conversation or a large tool result can blow the model's token
    # budget. Two compression points - history summarization and tool-output
    # compression - both run through one `Condenser`, and both are driven by a
    # token count. The condenser uses its own model because Groq applies its
    # tokens-per-minute limit per model: routing condensation to a cheap model
    # does not spend the chat model's budget.
    agent_condenser_model: str = "llama-3.1-8b-instant"
    # A hard slice on the input before a condenser call, so a pathological page
    # cannot blow the condenser's own budget. A safety ceiling, not the tuning
    # knob.
    agent_condenser_max_chars: int = 40_000
    # Summarize the thread once it passes this many tokens. Defaults are chosen
    # so a full reply - ~4,000 history + ~1,000 tool result + ~400 tool specs,
    # resent three times - sits under the free tier's budget.
    agent_history_token_budget: int = 4_000
    # How much of the thread is kept verbatim after a summary, so the model can
    # still answer about the most recent exchange without it being compressed.
    agent_recent_token_budget: int = 1_500
    # Compress a single tool result above this many tokens. A three-line
    # retrieval must not cost an extra model call.
    agent_tool_output_token_budget: int = 1_000
    # The instruction for history summarization.
    agent_summary_prompt: str = (
        "Summarize the conversation so far into a concise summary that preserves "
        "the key facts, decisions, the user's questions and preferences, and any "
        "URLs or names mentioned. Keep it under 200 words."
    )
    # The instruction for query-focused tool-output compression.
    agent_tool_condense_prompt: str = (
        "You are given a tool result and a focus. Rewrite the tool result to keep "
        "only the information relevant to the focus, preserving exact names, "
        "numbers, URLs and quotes. Be concise and do not invent anything."
    )

    # --- Tavily (web search) ---
    tavily_api_key: SecretStr
    tavily_max_results: int = 5

    # --- fetch_web_pages tool ---
    # `fetch_web_max_urls_per_call` tightens the tool's own hard ceiling
    # (`MAX_URLS_PER_CALL` in fetch_web_pages.py) rather than replacing it - that
    # ceiling stops a hallucinated list of URLs from becoming dozens of outbound
    # requests, and stays fixed regardless of this setting. This one is the
    # dial for trading off context size against how many pages one turn can read.
    fetch_web_max_urls_per_call: int = 10
    # Per-page character cap when `fetch_all` joins several pages into one blob
    # for a chat turn (`MAX_PAGE_CHARS` in aiohttp_scraper.py otherwise).
    fetch_web_max_page_chars: int = 20_000

    # --- Gemini (embeddings) ---
    gemini_api_key: SecretStr
    embedding_model: str = "gemini-embedding-2"
    embedding_dimensions: int = 768

    # --- Qdrant (vector store) ---
    # A full URL rather than host + port: Qdrant Cloud hands out an HTTPS
    # endpoint on 6333 with a managed certificate, and a host/port pair cannot
    # express the scheme. Local Docker is the same field, spelled http.
    qdrant_url: str = "http://localhost:6333"
    # Required by Qdrant Cloud, absent locally — hence optional rather than a
    # empty-string default that would be sent as a real (and wrong) key.
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "knowledge_base"
    retrieval_limit: int = 3
    retrieval_score_threshold: float = 0.7

    # --- Deepgram (speech to text) ---
    deepgram_api_key: SecretStr
    deepgram_model: str = "nova-3"
    utterance_end_ms: int = 1500
    finalize_timeout_seconds: float = 3.0

    # --- Authentication ---
    # No default, deliberately. A signing secret with a fallback is a signing
    # secret every deployment that forgot to set one shares, and anyone holding
    # it can mint a token for any account. Missing it fails at startup.
    jwt_secret: SecretStr
    jwt_algorithm: str = "HS256"
    # Short, because an access token cannot be withdrawn before it expires -
    # verification never touches the database, which is the point of it. Fifteen
    # minutes is the window a signed-out or disabled account keeps working.
    access_token_ttl_seconds: int = 15 * 60
    # Long, because this one *is* revocable: it lives as a row that rotation and
    # sign-out can mark dead.
    refresh_token_ttl_seconds: int = 14 * 24 * 60 * 60
    # How long a refresh token row outlives its own `expires_at` before the
    # cleanup job deletes it. Not zero: `RefreshSession` treats a *revoked* row
    # turning up again as reuse and revokes the whole family, and that signal is
    # only worth anything while the row still exists. A real replay attempt
    # shows up within days of rotation, not a month later, so 30 days keeps that
    # detection window generous while still bounding table growth.
    refresh_token_cleanup_retention_seconds: int = 30 * 24 * 60 * 60
    auth_rate_limit_attempts: int = 5
    auth_rate_limit_window_seconds: int = 5 * 60
    # Whether the caller's IP can be believed. Behind a proxy that does not set
    # forwarded headers, every request appears to come from the proxy, and an
    # IP-keyed limit would throttle every user at once the moment one of them
    # mistyped a password. Off unless the deployment says otherwise.
    trust_forwarded_client_ip: bool = False

    refresh_cookie_name: str = "refresh_token"
    # Scoped to the auth routes, so the cookie is not sent on every chat request
    # and every upload. It is only ever needed by refresh and sign-out.
    refresh_cookie_path: str = "/auth"
    refresh_cookie_secure: bool = True
    # "none" because the frontend and the API are on different sites in the
    # deployed topology (Vercel and the API domain), and a "lax" cookie is simply
    # not sent there. Local development runs same-origin through the Vite proxy,
    # where "lax" is both sufficient and what a plain-http origin will accept -
    # `SameSite=None` requires `Secure`, which requires https.
    refresh_cookie_samesite: str = "none"
    refresh_cookie_domain: str | None = None

    # --- Spend limits ---
    # Separate from the auth limit above, and set for a different reason. That
    # one exists to stop password guessing; these exist because every chat reply
    # and every accepted upload spends real money at Groq and Gemini, and an
    # account left looping costs whoever pays the bill, not the account.
    #
    # Deliberately tight - tight enough that a fast reader will meet them - and
    # tight because the budget behind this deployment is small. Raise them only
    # against a bill somebody has agreed to pay. Both are counted per signed-in
    # user.
    chat_rate_limit_requests: int = 3
    chat_rate_limit_window_seconds: int = 60
    upload_rate_limit_requests: int = 2
    upload_rate_limit_window_seconds: int = 10 * 60

    # One ceiling shared by every account, on top of the per-user ones above.
    # Registration has no CAPTCHA, so the per-user limits alone are only a limit
    # on how fast any *one* account can spend - someone willing to sign up
    # repeatedly can still multiply that by as many accounts as they create.
    # This is the backstop: one counter, one key, hit by chat replies, uploads,
    # URL ingestion and transcription sessions alike, so the total spend across
    # every caller on this deployment cannot exceed what the default 200/day
    # covers regardless of how many accounts asked for it.
    global_daily_call_budget: int = 200
    global_daily_call_budget_window_seconds: int = 24 * 60 * 60

    # --- Ingestion ---
    upload_dir: Path = Path("uploads")
    static_pages_dir: Path = Path("pages")
    max_upload_bytes: int = 25 * 1024 * 1024
    chunk_size: int = 500
    chunk_overlap: int = 50

    @model_validator(mode="after")
    def _default_websocket_origins_to_cors(self) -> "Settings":
        """Fall back to the CORS list, so one variable configures both.

        Done here rather than at the point of use so that every reader sees the
        resolved list, and so a deployment that genuinely wants them to differ
        can still say so by setting the variable.
        """
        if not self.allowed_websocket_origins:
            self.allowed_websocket_origins = list(self.cors_origins)
        return self

    @property
    def migration_url(self) -> str:
        """The URL migrations connect with — the direct endpoint when there is one."""
        return self.database_migration_url or self.database_url


@lru_cache
def get_settings() -> Settings:
    """Read the environment once. Called from `create_app` and nowhere else."""
    return Settings()
