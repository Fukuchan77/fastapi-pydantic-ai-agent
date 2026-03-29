"""Tests for _get_cached_model cache invalidation strategy.

Task 30.1: The @lru_cache decorator on _get_cached_model() doesn't key the cache
by settings values. When settings change (e.g., LLM_MODEL environment variable),
the cached model persists instead of being rebuilt with the new settings.

This test demonstrates the issue and verifies the fix.
"""

from app.deps.workflow import _get_cached_model


def test_get_cached_model_invalidates_on_settings_change(monkeypatch):
    """Test that _get_cached_model returns new model when settings change.

    Bug: The current @lru_cache(maxsize=1) implementation caches based on
    function arguments (none), so the cache key is always (). This means
    changing environment variables and clearing get_settings() cache is
    insufficient - the old model persists.

    Expected behavior: When LLM_MODEL changes, _get_cached_model should
    return a new model instance configured with the new model name.
    """
    from app.config import get_settings

    # Clear all caches before test
    get_settings.cache_clear()
    _get_cached_model.cache_clear()

    # Step 1: Set initial model configuration
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key-12345")
    monkeypatch.setenv("API_KEY", "test-api-key-12345")

    # Get first model instance
    settings1 = get_settings()
    model1 = _get_cached_model(
        llm_model=settings1.llm_model,
        llm_base_url=settings1.llm_base_url,
    )

    # Verify initial configuration
    assert settings1.llm_model == "openai:gpt-4"

    # Step 2: Change settings to different model
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-3.5-turbo")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key-67890")

    # Clear get_settings cache to pick up new environment variables
    get_settings.cache_clear()

    # Verify settings have changed
    settings2 = get_settings()
    assert settings2.llm_model == "openai:gpt-3.5-turbo"
    assert settings2.llm_model != settings1.llm_model

    # Step 3: Get model again - should be NEW instance with new settings
    model2 = _get_cached_model(
        llm_model=settings2.llm_model,
        llm_base_url=settings2.llm_base_url,
    )

    # CRITICAL ASSERTION: Models should be DIFFERENT objects
    # because the settings changed
    assert model2 is not model1, (
        "Expected new model instance when settings change, but got cached model. "
        "The @lru_cache decorator doesn't detect settings changes."
    )

    # Verify the new model reflects the new settings
    # Both models are LiteLLMModel instances, but they should have different configs
    assert model1 != model2


def test_get_cached_model_returns_same_instance_when_settings_unchanged():
    """Test that _get_cached_model returns cached instance when settings don't change.

    This verifies that caching still works - we want to reuse models when
    settings haven't changed.
    """
    from app.config import get_settings

    # Clear caches
    get_settings.cache_clear()
    _get_cached_model.cache_clear()

    # Get model twice without changing settings
    settings = get_settings()
    model1 = _get_cached_model(
        llm_model=settings.llm_model,
        llm_base_url=settings.llm_base_url,
    )
    model2 = _get_cached_model(
        llm_model=settings.llm_model,
        llm_base_url=settings.llm_base_url,
    )

    # Should be same instance (cached)
    assert model1 is model2


def test_get_cached_model_cache_clear_forces_rebuild(monkeypatch):
    """Test that explicit cache_clear() on _get_cached_model rebuilds the model.

    This documents the workaround used in tests (conftest.py clear_workflow_cache).
    Even without settings changes, calling cache_clear() should force rebuild.
    """
    from app.config import get_settings

    # Clear caches
    get_settings.cache_clear()
    _get_cached_model.cache_clear()

    # Set settings
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key-12345")
    monkeypatch.setenv("API_KEY", "test-api-key-12345")

    # Get first model
    settings = get_settings()
    model1 = _get_cached_model(
        llm_model=settings.llm_model,
        llm_base_url=settings.llm_base_url,
    )

    # Force cache clear
    _get_cached_model.cache_clear()

    # Get model again - should be NEW instance even though settings unchanged
    model2 = _get_cached_model(
        llm_model=settings.llm_model,
        llm_base_url=settings.llm_base_url,
    )

    # Models should be different objects after explicit cache clear
    assert model1 is not model2
