"""Unit tests for workflow dependency functions."""

import pytest
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from pydantic_ai.models.test import TestModel

from tests.conftest import build_test_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    """Clear get_settings() cache before each test."""
    from app.config import get_settings

    get_settings.cache_clear()


def _make_request(app: FastAPI) -> Request:
    """Build a real `Request` bound to `app`, so unset `app.state` attrs raise `AttributeError`.

    A bare `Mock()` cannot represent this: its attribute access
    auto-vivifies, so an "absent" attribute would silently return another
    `Mock` instead of raising, like production `app.state` does.
    """
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "query_string": b"",
        "headers": [],
        "app": app,
    }
    return Request(scope=scope, receive=None)  # type: ignore[arg-type]


def test_get_rag_workflow_reads_settings_and_model_from_app_state() -> None:
    """Req 3.3: the workflow dependency must resolve settings and model from `app.state`.

    Not from process-global `get_settings()` or a model cache keyed on model
    name and base URL.

    RED today (Req 13.5): `get_rag_workflow` calls `get_settings()` for
    settings, so the distinguishable settings object below never reaches the
    workflow, and it builds a model via `_get_cached_model` (deleted later in
    this unit), so the distinguishable model below never reaches it either.
    """
    from app.deps.workflow import get_rag_workflow
    from app.stores.vector_store import InMemoryVectorStore

    distinguishable_settings = build_test_settings(
        cors_origins=["https://distinguishable.example"]
    )
    distinguishable_model = TestModel()

    app = FastAPI()
    app.state.vector_store = InMemoryVectorStore()
    app.state.settings = distinguishable_settings
    app.state.llm_model = distinguishable_model

    workflow = get_rag_workflow(_make_request(app))

    assert workflow.llm_settings is distinguishable_settings
    assert workflow.llm_model is distinguishable_model


def test_get_rag_workflow_503s_when_settings_singleton_absent() -> None:
    """Req 3.4: an absent `app.state.settings` must fail the request, never fall back.

    Fails in the flat envelope with `code="DEPENDENCY_NOT_INITIALIZED"`,
    never falling back to process-global settings.

    RED today (Req 13.5): `get_rag_workflow` falls back to `get_settings()`
    and the request succeeds instead of failing - precisely the bug this
    unit removes.
    """
    from app.deps.workflow import get_rag_workflow
    from app.stores.vector_store import InMemoryVectorStore

    app = FastAPI()
    app.state.vector_store = InMemoryVectorStore()
    app.state.llm_model = TestModel()
    # app.state.settings intentionally left unset.

    with pytest.raises(HTTPException) as exc_info:
        get_rag_workflow(_make_request(app))

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["code"] == "DEPENDENCY_NOT_INITIALIZED"


def test_get_rag_workflow_503s_when_model_singleton_absent() -> None:
    """Req 3.4, the `app.state.llm_model`-absent half of the same contract.

    RED today (Req 13.5): `get_rag_workflow` never reads `app.state.llm_model`
    at all - it builds its own model via `_get_cached_model` - so the request
    succeeds instead of failing.
    """
    from app.deps.workflow import get_rag_workflow
    from app.stores.vector_store import InMemoryVectorStore

    app = FastAPI()
    app.state.vector_store = InMemoryVectorStore()
    app.state.settings = build_test_settings()
    # app.state.llm_model intentionally left unset.

    with pytest.raises(HTTPException) as exc_info:
        get_rag_workflow(_make_request(app))

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["code"] == "DEPENDENCY_NOT_INITIALIZED"


def test_get_rag_workflow_returns_workflow_instance() -> None:
    """`get_rag_workflow` returns a `CorrectiveRAGWorkflow` instance.

    Reworked (Req 3.3, 3.7) to build the real model via `build_model()` and
    inject it through `app.state` - the deleted `_get_cached_model` this
    test used to exercise indirectly (via a bare `Mock()` request, whose
    auto-vivifying attributes could never represent the dependency's actual
    `app.state` contract) no longer exists.
    """
    from app.agents.chat_agent import build_model
    from app.deps.workflow import get_rag_workflow
    from app.stores.vector_store import InMemoryVectorStore
    from app.workflows.corrective_rag import CorrectiveRAGWorkflow

    settings = build_test_settings(llm_model="ollama:llama2", llm_base_url="http://localhost:11434")
    app = FastAPI()
    app.state.vector_store = InMemoryVectorStore()
    app.state.settings = settings
    app.state.llm_model = build_model(settings)

    workflow = get_rag_workflow(_make_request(app))

    assert isinstance(workflow, CorrectiveRAGWorkflow)


def test_get_rag_workflow_uses_app_state_vector_store() -> None:
    """`get_rag_workflow` uses the vector store from `app.state` (Req 3.3)."""
    from app.agents.chat_agent import build_model
    from app.deps.workflow import get_rag_workflow
    from app.stores.vector_store import InMemoryVectorStore

    settings = build_test_settings(llm_model="ollama:llama2", llm_base_url="http://localhost:11434")
    mock_vector_store = InMemoryVectorStore()
    app = FastAPI()
    app.state.vector_store = mock_vector_store
    app.state.settings = settings
    app.state.llm_model = build_model(settings)

    workflow = get_rag_workflow(_make_request(app))

    assert workflow.vector_store is mock_vector_store


def test_get_rag_workflow_uses_app_state_settings() -> None:
    """`get_rag_workflow` passes `app.state.settings` through to the workflow (Req 3.3).

    Not process-global `get_settings()`.
    """
    from app.agents.chat_agent import build_model
    from app.config import Settings
    from app.deps.workflow import get_rag_workflow
    from app.stores.vector_store import InMemoryVectorStore

    settings = build_test_settings(llm_model="ollama:llama2", llm_base_url="http://localhost:11434")
    app = FastAPI()
    app.state.vector_store = InMemoryVectorStore()
    app.state.settings = settings
    app.state.llm_model = build_model(settings)

    workflow = get_rag_workflow(_make_request(app))

    assert isinstance(workflow.llm_settings, Settings)
    assert workflow.llm_settings is settings
