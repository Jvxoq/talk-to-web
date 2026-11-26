"""`DATABASE_URL` and `DATABASE_MIGRATION_URL` must name one database."""

from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from app.settings import Settings

POOLED = "postgresql+psycopg://app:app-secret@ep-cool-123-pooler.eu-west-2.aws.neon.tech/neondb"
DIRECT = "postgresql+psycopg://app:app-secret@ep-cool-123.eu-west-2.aws.neon.tech/neondb"


def build(database_url: str, migration_url: str | None, **overrides: Any) -> Settings:
    """Explicit dummies and no .env, so a developer's machine cannot change a result."""
    return Settings(
        _env_file=None,
        environment="test",
        database_url=database_url,
        database_migration_url=migration_url,
        llm_api_key=SecretStr("test"),
        tavily_api_key=SecretStr("test"),
        gemini_api_key=SecretStr("test"),
        deepgram_api_key=SecretStr("test"),
        jwt_secret=SecretStr("test"),
        **overrides,
    )


def test_unset_migration_url_skips_check() -> None:
    settings = build(POOLED, None)

    assert settings.migration_url == POOLED


def test_identical_urls_pass() -> None:
    settings = build(DIRECT, DIRECT)

    assert settings.migration_url == DIRECT


def test_pooler_and_direct_host_pass() -> None:
    settings = build(POOLED, DIRECT)

    assert settings.migration_url == DIRECT


def test_direct_app_url_with_pooler_migration_passes() -> None:
    settings = build(DIRECT, POOLED)

    assert settings.migration_url == POOLED


def test_driver_and_query_differences_pass() -> None:
    plain = DIRECT.replace("postgresql+psycopg", "postgresql") + "?sslmode=require"

    settings = build(POOLED, plain)

    assert settings.migration_url == plain


def test_default_port_matches_explicit_port() -> None:
    settings = build(DIRECT, DIRECT.replace(".tech/", ".tech:5432/"))

    assert settings.migration_url.endswith(".tech:5432/neondb")


def test_different_role_passes() -> None:
    """Migrating as the role that owns the tables is a normal setup."""
    owner = DIRECT.replace("app:app-secret", "owner:owner-secret")

    settings = build(POOLED, owner)

    assert settings.migration_url == owner


def test_different_endpoint_fails() -> None:
    other = DIRECT.replace("ep-cool-123", "ep-other-456")

    with pytest.raises(
        ValidationError,
        match=r"DATABASE_URL names ep-cool-123-pooler\.eu-west-2\.aws\.neon\.tech:5432/neondb "
        r"but DATABASE_MIGRATION_URL names ep-other-456\.eu-west-2\.aws\.neon\.tech:5432/neondb",
    ):
        build(POOLED, other)


def test_different_database_name_fails() -> None:
    with pytest.raises(ValidationError, match=r"neondb but DATABASE_MIGRATION_URL names .*/other"):
        build(POOLED, DIRECT.replace("/neondb", "/other"))


def test_different_port_fails() -> None:
    with pytest.raises(ValidationError, match=r"DATABASE_MIGRATION_URL names .*:6543/neondb"):
        build(POOLED, DIRECT.replace(".tech/", ".tech:6543/"))


def test_pooler_inside_later_label_fails() -> None:
    """Only the first host label may carry the suffix."""
    other = DIRECT.replace("eu-west-2", "eu-west-2-pooler")

    with pytest.raises(ValidationError, match="must be the same database"):
        build(DIRECT, other)


@pytest.mark.parametrize(
    "migration_url",
    [DIRECT.replace("ep-cool-123", "ep-other-456"), DIRECT.replace("/neondb", "/other")],
    ids=["other-host", "other-database"],
)
def test_error_never_prints_passwords(migration_url: str) -> None:
    migration_url = migration_url.replace("app-secret", "migration-secret")

    with pytest.raises(ValidationError) as caught:
        build(POOLED, migration_url)

    message = str(caught.value)
    assert "app-secret" not in message
    assert "migration-secret" not in message


def test_unparseable_url_error_never_prints_password() -> None:
    with pytest.raises(ValidationError, match="DATABASE_MIGRATION_URL is not a valid") as caught:
        build(POOLED, "postgresql://app:migration-secret@host:notaport/neondb")

    assert "migration-secret" not in str(caught.value)


def build_llm(**llm: Any) -> Settings:
    """`build`, with only the LLM variables under test given."""
    return Settings(
        _env_file=None,
        environment="test",
        database_url=DIRECT,
        llm_api_key=SecretStr("test"),
        tavily_api_key=SecretStr("test"),
        gemini_api_key=SecretStr("test"),
        deepgram_api_key=SecretStr("test"),
        jwt_secret=SecretStr("test"),
        **llm,
    )


def test_default_provider_uses_the_gateway() -> None:
    assert build_llm().llm_base_url == "https://opencode.ai/inference/openai/v1"


def test_switched_provider_without_url_uses_its_own_endpoint() -> None:
    assert build_llm(llm_provider="anthropic").llm_base_url is None


def test_switched_provider_keeps_an_explicit_url() -> None:
    settings = build_llm(llm_provider="ollama", llm_base_url="http://localhost:11434")

    assert settings.llm_base_url == "http://localhost:11434"


def test_empty_base_url_means_the_providers_own_endpoint() -> None:
    assert build_llm(llm_base_url="").llm_base_url is None


@pytest.mark.parametrize("field", ["llm_request_timeout_seconds", "chat_reply_timeout_seconds"])
def test_zero_timeout_is_refused(field: str) -> None:
    """Zero reads as "no timeout" to some clients and fires at once in asyncio."""
    with pytest.raises(ValidationError, match="greater than 0"):
        build(DIRECT, None, **{field: 0})
