"""Model-level proof that the chat ceiling/temperature reach every model tried.

`test_chat_agent.py::TestBuildChatAgent` pins the Agent-construction contract
(`agent.model_settings`); this file proves the value survives all the way to
the post-merge `AgentInfo.model_settings` a `FunctionModel` actually receives
- once for the primary model, and once for a fallback member selected after
the primary fails (Req 9.7).
"""

from unittest.mock import Mock

import httpx
import pytest
from pydantic_ai import ModelHTTPError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings

from app.agents.chat_agent import build_chat_agent
from app.agents.deps import AgentDeps
from app.stores.session_store import SessionStore
from tests.conftest import build_test_settings


# supports_json_schema_output=False keeps build_chat_agent on the plain-str
# output path, so a bare TextPart response is accepted without NativeOutput
# schema parsing - orthogonal to what this file is proving.
_PLAIN_TEXT_PROFILE = ModelProfile(supports_json_schema_output=False)


def _build_deps() -> AgentDeps:
    """Minimal deps - the run never calls a tool, so these are never touched."""
    return AgentDeps(
        http_client=Mock(spec=httpx.AsyncClient),
        settings=build_test_settings(),
        session_store=Mock(spec=SessionStore),
    )


class TestModelSettingsReachThePrimaryModel:
    """Req 9.1/9.2: the ceiling and temperature reach the primary model's request."""

    @pytest.mark.asyncio
    async def test_reaches_agent_info_for_a_lone_primary_model(self) -> None:
        """A single-member FallbackModel mirrors build_fallback_model's chain-of-one."""
        seen: list[ModelSettings | None] = []

        def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(info.model_settings)
            return ModelResponse(parts=[TextPart(content="primary answered")])

        primary = FunctionModel(respond, model_name="primary", profile=_PLAIN_TEXT_PROFILE)
        chain = FallbackModel(primary)
        settings = build_test_settings(llm_max_output_tokens=2048, llm_temperature=0.15)
        agent = build_chat_agent(model=chain, settings=settings)

        result = await agent.run("hi", deps=_build_deps())

        assert seen == [{"max_tokens": 2048, "temperature": 0.15}]
        assert result.output == "primary answered"


class TestModelSettingsReachASelectedFallbackModel:
    """Req 9.3/9.7: the same values reach a fallback member chosen after the primary fails."""

    @pytest.mark.asyncio
    async def test_reaches_agent_info_for_both_the_failed_primary_and_the_fallback(
        self,
    ) -> None:
        """Both chain members must observe the identical post-merge settings dict.

        The primary's function records what it received before raising
        `ModelHTTPError` (the default `fallback_on=(ModelAPIError,)` trigger),
        so this proves the value reached the primary attempt too - not just
        the member that ultimately answered.
        """
        primary_seen: list[ModelSettings | None] = []
        fallback_seen: list[ModelSettings | None] = []

        def failing_primary(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            primary_seen.append(info.model_settings)
            raise ModelHTTPError(status_code=500, model_name="primary")

        def succeeding_fallback(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            fallback_seen.append(info.model_settings)
            return ModelResponse(parts=[TextPart(content="fallback answered")])

        primary = FunctionModel(failing_primary, model_name="primary", profile=_PLAIN_TEXT_PROFILE)
        fallback = FunctionModel(
            succeeding_fallback, model_name="fallback", profile=_PLAIN_TEXT_PROFILE
        )
        chain = FallbackModel(primary, fallback)
        settings = build_test_settings(llm_max_output_tokens=2048, llm_temperature=0.15)
        agent = build_chat_agent(model=chain, settings=settings)

        result = await agent.run("hi", deps=_build_deps())

        expected: ModelSettings = {"max_tokens": 2048, "temperature": 0.15}
        assert primary_seen == [expected]
        assert fallback_seen == [expected]
        assert result.output == "fallback answered"
