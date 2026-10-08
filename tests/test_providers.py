import pytest


def test_ollama_is_default_and_targets_localhost() -> None:
    from langchain_ollama import ChatOllama
    from open_rag.providers import make_model

    model = make_model("ollama")
    assert isinstance(model, ChatOllama)
    assert model.model == "phi4-mini"
    assert model.base_url == "http://127.0.0.1:11434"


def test_openai_requires_environment_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from langchain_openai import ChatOpenAI
    from open_rag.providers import ProviderError, make_model

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        make_model("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    model = make_model("openai")
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-4o-mini"


def test_unknown_provider_is_rejected() -> None:
    from open_rag.providers import ProviderError, make_model

    with pytest.raises(ProviderError, match="Unknown provider"):
        make_model("claude")
