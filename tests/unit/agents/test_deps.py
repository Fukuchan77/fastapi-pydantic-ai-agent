"""Unit tests for agent dependencies module."""

from dataclasses import fields
from unittest.mock import Mock

import httpx
import pytest

from app.agents.deps import AgentDeps
from app.agents.deps import get_agent_deps
from app.config import Settings
from app.stores.session_store import SessionStore


class TestAgentDeps:
    """Test suite for AgentDeps dataclass."""

    def test_agent_deps_is_dataclass(self) -> None:
        """AgentDeps should be a dataclass."""
        assert hasattr(AgentDeps, "__dataclass_fields__")

    def test_agent_deps_has_required_fields(self) -> None:
        """AgentDeps should have http_client, settings, and session_store fields."""
        field_names = {f.name for f in fields(AgentDeps)}
        expected_fields = {"http_client", "settings", "session_store"}
        assert field_names == expected_fields

    def test_agent_deps_field_types(self) -> None:
        """AgentDeps fields should have correct type annotations."""
        field_dict = {f.name: f.type for f in fields(AgentDeps)}

        assert field_dict["http_client"] == httpx.AsyncClient
        assert field_dict["settings"] == Settings
        assert field_dict["session_store"] == SessionStore

    def test_agent_deps_construction(self) -> None:
        """AgentDeps should be constructable with all required fields."""
        mock_client = Mock(spec=httpx.AsyncClient)
        mock_settings = Mock(spec=Settings)
        mock_store = Mock(spec=SessionStore)

        deps = AgentDeps(
            http_client=mock_client,
            settings=mock_settings,
            session_store=mock_store,
        )

        assert deps.http_client is mock_client
        assert deps.settings is mock_settings
        assert deps.session_store is mock_store

    def test_agent_deps_immutable_after_frozen(self) -> None:
        """AgentDeps fields should be accessible after construction."""
        mock_client = Mock(spec=httpx.AsyncClient)
        mock_settings = Mock(spec=Settings)
        mock_store = Mock(spec=SessionStore)

        deps = AgentDeps(
            http_client=mock_client,
            settings=mock_settings,
            session_store=mock_store,
        )

        # Fields should be readable
        assert isinstance(deps.http_client, Mock)
        assert isinstance(deps.settings, Mock)
        assert isinstance(deps.session_store, Mock)


class TestGetAgentDeps:
    """Test suite for get_agent_deps FastAPI dependency."""

    @pytest.mark.asyncio
    async def test_get_agent_deps_extracts_from_app_state(self) -> None:
        """get_agent_deps should extract dependencies from request.app.state."""
        # Arrange
        mock_client = Mock(spec=httpx.AsyncClient)
        mock_settings = Mock(spec=Settings)
        mock_store = Mock(spec=SessionStore)

        mock_request = Mock()
        mock_request.app.state.http_client = mock_client
        mock_request.app.state.settings = mock_settings
        mock_request.app.state.session_store = mock_store

        # Act
        deps = await get_agent_deps(mock_request)

        # Assert
        assert isinstance(deps, AgentDeps)
        assert deps.http_client is mock_client
        assert deps.settings is mock_settings
        assert deps.session_store is mock_store

    @pytest.mark.asyncio
    async def test_get_agent_deps_returns_agent_deps_instance(self) -> None:
        """get_agent_deps should return an AgentDeps instance."""
        # Arrange
        mock_request = Mock()
        mock_request.app.state.http_client = Mock(spec=httpx.AsyncClient)
        mock_request.app.state.settings = Mock(spec=Settings)
        mock_request.app.state.session_store = Mock(spec=SessionStore)

        # Act
        deps = await get_agent_deps(mock_request)

        # Assert
        assert isinstance(deps, AgentDeps)
        assert hasattr(deps, "http_client")
        assert hasattr(deps, "settings")
        assert hasattr(deps, "session_store")
