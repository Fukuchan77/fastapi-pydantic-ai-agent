"""Unit tests: /v1/agent/chat correctly unwraps NativeOutput(ChatOutput) (Task 7.3).

When the active model's profile supports JSON-schema output, `build_chat_agent`
wraps its output in `NativeOutput(ChatOutput)` (Req 10.2) and `result.output`
becomes a `ChatOutput` instance instead of `str`. This must not leak into the
`ChatResponse.reply` field as a stringified Pydantic object.
"""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.profiles import ModelProfile

from app.agents.chat_agent import build_chat_agent
from app.agents.deps import AgentDeps
from app.agents.deps import get_agent_deps
from app.agents.guardrails import AuditTrail
from app.api.v1.agent import router
from app.deps.auth import verify_api_key
from app.middleware.rate_limit import enforce_llm_rate_limit
from app.security.principal import Principal
from tests.conftest import build_test_settings


@pytest.fixture
def mock_agent_deps() -> AgentDeps:
    """Create mock AgentDeps for testing."""
    deps = MagicMock(spec=AgentDeps)
    deps.session_store = MagicMock()
    deps.session_store.get_history = AsyncMock(return_value=[])
    deps.session_store.save_history = AsyncMock()
    deps.settings = build_test_settings()
    deps.audit = AuditTrail()
    return deps


def _build_app(mock_agent_deps: AgentDeps, chat_agent: object) -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    app.state.chat_agent = chat_agent
    app.state.settings = build_test_settings()
    app.dependency_overrides[get_agent_deps] = lambda: mock_agent_deps
    app.dependency_overrides[verify_api_key] = lambda: Principal(id="test-principal")
    app.dependency_overrides[enforce_llm_rate_limit] = lambda: None
    return app


def _native_output_model() -> FunctionModel:
    """A FunctionModel whose profile reports JSON-schema support.

    Returns raw JSON text matching the ChatOutput schema.
    """

    def _reply_json(messages: list, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='{"reply": "Hello there"}')])

    return FunctionModel(
        _reply_json,
        profile=ModelProfile(supports_json_schema_output=True),
    )


def test_chat_endpoint_unwraps_native_output_reply(mock_agent_deps: AgentDeps) -> None:
    """The ChatResponse.reply field must be the plain reply text, not a stringified object."""
    chat_agent = build_chat_agent(model=_native_output_model())
    app = _build_app(mock_agent_deps, chat_agent)
    client = TestClient(app)

    response = client.post("/v1/agent/chat", json={"message": "Hi"})

    assert response.status_code == 200
    assert response.json()["reply"] == "Hello there"
