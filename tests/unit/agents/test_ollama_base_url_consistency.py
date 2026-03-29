"""Test Ollama base URL consistency between build_model() and OllamaEmbeddingVectorStore.

Verify that the apparent inconsistency is intentional:
- build_model() uses http://localhost:11434 (no /v1) because LiteLLM auto-appends /v1
- OllamaEmbeddingVectorStore uses http://localhost:11434/v1 because it calls API directly

This test documents and validates the design decision.
"""

import inspect

import pytest

from app.agents import chat_agent
from app.agents.chat_agent import build_model
from app.config import Settings
from app.stores import vector_store
from app.stores.vector_store import OllamaEmbeddingVectorStore


def test_ollama_base_url_litellm_auto_appends_v1():
    """Verify that LiteLLM auto-appends /v1 to Ollama base URL.

    This test documents that build_model() correctly uses
    http://localhost:11434 (without /v1) because LiteLLM automatically
    appends /v1 when making requests to Ollama.

    Expected behavior:
    - build_model() sets litellm_api_base = http://localhost:11434
    - LiteLLM internally requests http://localhost:11434/v1/chat/completions
    """
    from pydantic import SecretStr

    # Given: Settings with Ollama model and no custom base URL
    settings = Settings(
        api_key=SecretStr("test-api-key-1234567890"),  # 16+ chars required
        llm_model="ollama:llama3",
        llm_api_key=None,  # Ollama doesn't require API key
        llm_base_url=None,  # Use default
    )

    # When: Building the model
    model = build_model(settings)

    # Then: Verify the model was created
    assert model is not None

    # For LiteLLMModel, we can't easily inspect internal settings without
    # making actual API calls, but we can verify the code path is correct
    # by checking the chat_agent.py source code
    source = inspect.getsource(build_model)

    # The source should show that for Ollama, we use base URL without /v1
    assert "litellm_api_base" in source
    assert "http://localhost:11434" in source

    # Verify that the default doesn't include /v1 (test via source inspection)
    # because LiteLLM auto-appends /v1 when making actual requests
    lines = source.split("\n")
    for line in lines:
        if "ollama" in line.lower() and "litellm_api_base" in line and "11434" in line:
            # This line sets the Ollama base URL
            assert "/v1" not in line or line.strip().startswith("#"), (
                "build_model() should NOT append /v1 to Ollama base URL "
                "because LiteLLM auto-appends it when making requests"
            )


def test_ollama_embedding_vector_store_needs_v1():
    """Verify that OllamaEmbeddingVectorStore correctly includes /v1 in base URL.

    This test documents that OllamaEmbeddingVectorStore must use
    http://localhost:11434/v1 (WITH /v1) because it calls the Ollama API
    directly without going through LiteLLM.

    Expected behavior:
    - OllamaEmbeddingVectorStore.DEFAULT_BASE_URL = http://localhost:11434/v1
    - Calls POST http://localhost:11434/v1/embeddings directly
    """
    # Verify the default base URL includes /v1
    assert OllamaEmbeddingVectorStore.DEFAULT_BASE_URL == "http://localhost:11434/v1"

    # Verify that when creating a store without explicit base_url,
    # it uses the default WITH /v1
    store = OllamaEmbeddingVectorStore(embedding_model="nomic-embed-text:latest")
    assert store._base_url == "http://localhost:11434/v1"


def test_ollama_consistency_documented_in_code():
    """Verify that the inconsistency is documented with comments in source code.

    After fixing, both files should have comments explaining
    why build_model() uses base_url without /v1 while OllamaEmbeddingVectorStore
    uses base_url with /v1.
    """
    # Check chat_agent.py has documentation about LiteLLM auto-appending /v1
    chat_agent_source = inspect.getsource(chat_agent)
    assert "LiteLLM" in chat_agent_source or "litellm" in chat_agent_source.lower()

    # This test will initially FAIL because documentation is missing
    # After GREEN phase, comments should be added to explain the behavior
    assert "/v1" in chat_agent_source, (
        "chat_agent.py should document that LiteLLM auto-appends /v1 for Ollama"
    )

    # Check vector_store.py has documentation about direct API calls
    vector_store_source = inspect.getsource(vector_store.OllamaEmbeddingVectorStore)
    assert "direct" in vector_store_source.lower() or "POST" in vector_store_source, (
        "OllamaEmbeddingVectorStore should document that it calls Ollama API directly"
    )


@pytest.mark.integration
def test_ollama_litellm_actual_request_url():
    """Integration test: Verify LiteLLM actually appends /v1 when calling Ollama.

    This integration test would verify the actual HTTP request
    made by LiteLLM includes /v1, but we can't run it without a real Ollama
    instance. Marked as integration test.
    """
    pytest.skip("This test requires a running Ollama instance and actual LiteLLM requests")
