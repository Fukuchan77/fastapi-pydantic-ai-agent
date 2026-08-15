"""Tests that agent routes attribute their audit trail to a principal (Req 4.7).

`AgentDeps.principal` existed but was never populated: `get_agent_deps()` builds
deps from `app.state` alone, and no route set the field afterwards. Every
`AuditRecord` a guarded run produced therefore recorded *what* was refused but
never *for whom* - most of an audit trail's value once more than one API key
exists, and the field's own docstring described a condition ("until session
ownership resolves a real principal") that session ownership had already met.

Binding happens at the route rather than inside `get_agent_deps()` because
resolving the principal there would import `app.deps.auth` and close an import
cycle (`app.deps` -> `app.deps.workflow` -> `app.workflows.corrective_rag` ->
`app.agents.chat_agent` -> `app.agents.deps`). These tests pin the binding at
both routes so a future one is not silently added without it.
"""

from unittest.mock import Mock

import httpx
from fastapi import Request
from fastapi.testclient import TestClient
from pydantic_ai.models.function import FunctionModel

from app.agents.deps import AgentDeps
from app.agents.deps import bind_principal
from app.agents.deps import get_agent_deps
from app.config import Settings
from app.deps.auth import verify_api_key
from app.main import create_app
from app.middleware.rate_limit import enforce_llm_rate_limit
from app.security.principal import Principal
from app.stores.session_store import SessionStore
from tests.conftest import build_test_settings


PRINCIPAL_ID = "abc123def4567890"


class TestBindPrincipal:
    """The binding helper itself."""

    def test_sets_the_principal_on_the_instance(self) -> None:
        """Mutating in place is safe: deps are constructed per request."""
        deps = AgentDeps(
            http_client=Mock(spec=httpx.AsyncClient),
            settings=Mock(spec=Settings),
            session_store=Mock(spec=SessionStore),
        )
        assert deps.principal is None

        returned = bind_principal(deps, PRINCIPAL_ID)

        assert returned is deps
        assert deps.principal == PRINCIPAL_ID

    async def test_get_agent_deps_leaves_it_unbound(self) -> None:
        """The dependency deliberately does not resolve the principal itself."""
        request = Mock()
        request.app.state.http_client = Mock(spec=httpx.AsyncClient)
        request.app.state.settings = Mock(spec=Settings)
        request.app.state.session_store = Mock(spec=SessionStore)

        deps = await get_agent_deps(request)

        assert deps.principal is None


class TestRoutesBindThePrincipal:
    """Both agent routes must bind before the agent runs."""

    def _client_capturing_deps(
        self,
        test_model: FunctionModel,
        captured: dict[str, AgentDeps],
    ) -> TestClient:
        """Wire an app that records the `AgentDeps` each request was served with."""
        app = create_app(settings=build_test_settings(), model=test_model)

        async def capturing_deps(request: Request) -> AgentDeps:
            deps = await get_agent_deps(request)
            captured["deps"] = deps
            return deps

        app.dependency_overrides[get_agent_deps] = capturing_deps
        app.dependency_overrides[verify_api_key] = lambda: Principal(id=PRINCIPAL_ID)
        app.dependency_overrides[enforce_llm_rate_limit] = lambda: None
        return TestClient(app)

    def test_chat_route_binds_the_principal(self, test_model: FunctionModel) -> None:
        """`POST /v1/agent/chat` attributes its audit trail."""
        captured: dict[str, AgentDeps] = {}
        with self._client_capturing_deps(test_model, captured) as client:
            response = client.post("/v1/agent/chat", json={"message": "hello"})

        assert response.status_code == 200
        assert captured["deps"].principal == PRINCIPAL_ID

    def test_stream_route_binds_the_principal(self, test_model: FunctionModel) -> None:
        """`POST /v1/agent/stream` attributes its audit trail too.

        Covered separately because the two routes install their guarded toolset
        independently - a binding on one proves nothing about the other.
        """
        captured: dict[str, AgentDeps] = {}
        with self._client_capturing_deps(test_model, captured) as client:
            response = client.post("/v1/agent/stream", json={"message": "hello"})

        assert response.status_code == 200
        assert captured["deps"].principal == PRINCIPAL_ID
