import pytest

from medic import tracing
from medic.config import LLMSettings
from medic.llm import MissingApiKey, make_chat_model

KEY_VARS = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL",
    "LANGFUSE_HOST",
)


@pytest.fixture(autouse=True)
def no_keys(monkeypatch):
    for var in KEY_VARS:
        monkeypatch.delenv(var, raising=False)


def test_gemini_requires_its_own_variable(monkeypatch):
    with pytest.raises(MissingApiKey, match="GEMINI_API_KEY"):
        make_chat_model(LLMSettings(provider="gemini", model="gemini-2.5-flash"))
    # The provider library's default variable is deliberately not honoured.
    monkeypatch.setenv("GOOGLE_API_KEY", "not-used")
    with pytest.raises(MissingApiKey):
        make_chat_model(LLMSettings(provider="gemini", model="gemini-2.5-flash"))


def test_gemini_client_is_built_with_the_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    model = make_chat_model(LLMSettings(provider="gemini", model="gemini-2.5-flash"))
    assert type(model).__name__ == "ChatGoogleGenerativeAI"
    assert model.model.endswith("gemini-2.5-flash")


def test_unknown_provider():
    with pytest.raises(ValueError, match="unknown"):
        make_chat_model(LLMSettings(provider="ollama", model="x"))


def test_tracing_is_a_noop_without_keys():
    assert not tracing.tracing_enabled()
    assert tracing.make_callbacks() == []
    with tracing.trace("spike", run_results="x") as handle:
        assert handle.trace_id is None
        assert handle.url is None
    tracing.flush()


def test_base_url_accepts_old_and_new_names(monkeypatch):
    assert tracing.base_url() == tracing.DEFAULT_BASE_URL
    monkeypatch.setenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
    assert tracing.base_url() == "https://us.cloud.langfuse.com"
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://example.test")
    assert tracing.base_url() == "https://example.test"
    assert tracing.TraceHandle("abc").url == "https://example.test/trace/abc"
