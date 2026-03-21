"""Unit tests for app/agents/deps.py - AgentDeps dataclass and factory."""

from dataclasses import is_dataclass
from unittest.mock import Mock

import httpx
import pytest

from app.agents.deps import AgentDeps
from app.agents.deps import get_agent_deps
from app.config import Settings
from app.stores.session_store import SessionStore


class TestAgentDeps:
    """Tests for AgentDeps dataclass."""

    def test_agent_deps_is_dataclass(self):
        """AgentDeps must be a dataclass."""
        assert is_dataclass(AgentDeps)

    def test_agent_deps_has_http_client_field(self):
        """AgentDeps must have http_client field."""
        assert "http_client" in AgentDeps.__dataclass_fields__

    def test_agent_deps_has_settings_field(self):
        """AgentDeps must have settings field."""
        assert "settings" in AgentDeps.__dataclass_fields__

    def test_agent_deps_has_session_store_field(self):
        """AgentDeps must have session_store field."""
        assert "session_store" in AgentDeps.__dataclass_fields__

    def test_agent_deps_http_client_type_annotation(self):
        """http_client field must be typed as httpx.AsyncClient."""
        field = AgentDeps.__dataclass_fields__["http_client"]
        assert field.type == httpx.AsyncClient

    def test_agent_deps_settings_type_annotation(self):
        """Settings field must be typed as Settings."""
        field = AgentDeps.__dataclass_fields__["settings"]
        assert field.type == Settings

    def test_agent_deps_session_store_type_annotation(self):
        """session_store field must be typed as SessionStore."""
        field = AgentDeps.__dataclass_fields__["session_store"]
        # SessionStore is a Protocol, so check the annotation name
        assert "SessionStore" in str(field.type)

    def test_agent_deps_construction(self):
        """AgentDeps can be constructed with required fields."""
        client = Mock(spec=httpx.AsyncClient)
        settings = Mock(spec=Settings)
        store = Mock(spec=SessionStore)

        deps = AgentDeps(
            http_client=client,
            settings=settings,
            session_store=store,
        )

        assert deps.http_client is client
        assert deps.settings is settings
        assert deps.session_store is store


class TestGetAgentDeps:
    """Tests for get_agent_deps FastAPI dependency factory."""

    @pytest.mark.asyncio
    async def test_get_agent_deps_returns_agent_deps(self):
        """get_agent_deps must return an AgentDeps instance."""
        mock_request = Mock()
        mock_request.app.state.http_client = Mock(spec=httpx.AsyncClient)
        mock_request.app.state.settings = Mock(spec=Settings)
        mock_request.app.state.session_store = Mock(spec=SessionStore)

        result = await get_agent_deps(mock_request)

        assert isinstance(result, AgentDeps)

    @pytest.mark.asyncio
    async def test_get_agent_deps_uses_app_state_http_client(self):
        """get_agent_deps must use http_client from app.state."""
        mock_client = Mock(spec=httpx.AsyncClient)
        mock_request = Mock()
        mock_request.app.state.http_client = mock_client
        mock_request.app.state.settings = Mock(spec=Settings)
        mock_request.app.state.session_store = Mock(spec=SessionStore)

        result = await get_agent_deps(mock_request)

        assert result.http_client is mock_client

    @pytest.mark.asyncio
    async def test_get_agent_deps_uses_app_state_settings(self):
        """get_agent_deps must use settings from app.state."""
        mock_settings = Mock(spec=Settings)
        mock_request = Mock()
        mock_request.app.state.http_client = Mock(spec=httpx.AsyncClient)
        mock_request.app.state.settings = mock_settings
        mock_request.app.state.session_store = Mock(spec=SessionStore)

        result = await get_agent_deps(mock_request)

        assert result.settings is mock_settings

    @pytest.mark.asyncio
    async def test_get_agent_deps_uses_app_state_session_store(self):
        """get_agent_deps must use session_store from app.state."""
        mock_store = Mock(spec=SessionStore)
        mock_request = Mock()
        mock_request.app.state.http_client = Mock(spec=httpx.AsyncClient)
        mock_request.app.state.settings = Mock(spec=Settings)
        mock_request.app.state.session_store = mock_store

        result = await get_agent_deps(mock_request)

        assert result.session_store is mock_store
