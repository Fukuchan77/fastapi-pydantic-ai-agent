"""Unit tests for the tool-design reference implementation (Req 15, X-6).

Exercises the four conventions from `docs/tool-design-conventions.md` as
applied together in `app/agents/examples/tool_design_reference.py`, so the
reference stays correct as pydantic-ai/pydantic evolve rather than rotting
silently as unlinted, untested prose.
"""

from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr
from pydantic_ai import RunContext
from pydantic_ai import RunUsage
from pydantic_ai.models.test import TestModel

from app.agents.deps import AgentDeps
from app.agents.examples.tool_design_reference import DirectoryDetail
from app.agents.examples.tool_design_reference import DirectorySummary
from app.agents.examples.tool_design_reference import directory_get
from app.agents.examples.tool_design_reference import directory_search
from app.config import Settings


def _build_ctx() -> RunContext[AgentDeps]:
    deps = AgentDeps(
        http_client=MagicMock(),
        settings=Settings(
            api_key=SecretStr("test-api-key-1234567890"),
            llm_model="openai:gpt-4o",
            llm_api_key=SecretStr("test-llm-key-1234567890"),
        ),
        session_store=MagicMock(),
    )
    return RunContext(deps=deps, model=TestModel(), usage=RunUsage())


class TestDirectorySearchNaming:
    """Convention 1: `<resource>_<verb>` naming."""

    def test_functions_share_the_directory_resource_prefix(self) -> None:
        """Both reference tools share the `directory_` resource prefix."""
        assert directory_search.__name__ == "directory_search"
        assert directory_get.__name__ == "directory_get"


class TestDirectorySearchPagination:
    """Convention 2: pagination via `next_offset`."""

    @pytest.mark.asyncio
    async def test_first_page_reports_next_offset_when_more_remain(self) -> None:
        """A partial page reports the offset to resume from."""
        result = await directory_search(_build_ctx(), limit=2)

        assert result.total == 3
        assert len(result.items) == 2
        assert result.next_offset == 2

    @pytest.mark.asyncio
    async def test_last_page_reports_next_offset_none(self) -> None:
        """The final page reports `next_offset=None`, not an out-of-range offset."""
        result = await directory_search(_build_ctx(), offset=2, limit=2)

        assert len(result.items) == 1
        assert result.next_offset is None

    @pytest.mark.asyncio
    async def test_limit_is_clamped_to_the_max_ceiling(self) -> None:
        """An oversized limit is clamped, not rejected.

        Only 3 fixture records exist, so the ceiling never bites directly
        here, but the request for "all of them" in one page must still
        succeed rather than raise.
        """
        result = await directory_search(_build_ctx(), limit=999)

        assert result.total == 3
        assert result.next_offset is None

    @pytest.mark.asyncio
    async def test_query_filters_case_insensitively(self) -> None:
        """`query` matches regardless of the caller's casing."""
        result = await directory_search(_build_ctx(), query="MATHEMATICIAN")

        assert result.total == 2
        assert {item.name for item in result.items} == {"Ada Lovelace", "Alan Turing"}


class TestDirectorySearchResponseFormat:
    """Convention 3: `response_format` concise/detailed knob."""

    @pytest.mark.asyncio
    async def test_default_is_concise(self) -> None:
        """Omitting `response_format` yields the token-efficient concise view."""
        result = await directory_search(_build_ctx(), limit=1)

        assert isinstance(result.items[0], DirectorySummary)

    @pytest.mark.asyncio
    async def test_detailed_includes_role_and_notes(self) -> None:
        """`response_format="detailed"` includes the fields concise omits."""
        result = await directory_search(_build_ctx(), limit=1, response_format="detailed")

        item = result.items[0]
        assert isinstance(item, DirectoryDetail)
        assert item.role
        assert item.notes

    @pytest.mark.asyncio
    async def test_long_notes_are_truncated_in_detailed_view(self) -> None:
        """Detailed notes are capped so one record can't flood the context window."""
        result = await directory_get(_build_ctx(), "d-1", response_format="detailed")

        assert isinstance(result, DirectoryDetail)
        assert len(result.notes) <= 81  # _DETAIL_NOTE_CHARS + the ellipsis marker


class TestLenientParsing:
    """Convention 4: lenient argument parsing."""

    @pytest.mark.asyncio
    async def test_response_format_tolerates_case_and_whitespace(self) -> None:
        """Cosmetic variance in `response_format` doesn't fail the call."""
        result = await directory_search(_build_ctx(), limit=1, response_format=" Detailed ")

        assert isinstance(result.items[0], DirectoryDetail)

    @pytest.mark.asyncio
    async def test_unrecognized_response_format_falls_back_to_concise(self) -> None:
        """An unrecognized `response_format` value degrades to the default, not an error."""
        result = await directory_search(_build_ctx(), limit=1, response_format="verbose")

        assert isinstance(result.items[0], DirectorySummary)

    @pytest.mark.asyncio
    async def test_numeric_looking_string_limit_is_accepted(self) -> None:
        """A model-emitted numeric string for `limit` is coerced, not rejected."""
        result = await directory_search(_build_ctx(), limit="2")

        assert len(result.items) == 2

    @pytest.mark.asyncio
    async def test_malformed_limit_falls_back_to_default_instead_of_raising(self) -> None:
        """Genuinely malformed input degrades to the default rather than raising."""
        result = await directory_search(_build_ctx(), limit="not-a-number")

        assert len(result.items) == 3  # falls back to _DEFAULT_LIMIT (5), all 3 fit


class TestDirectoryGet:
    """`directory_get`: targeted single-record lookup."""

    @pytest.mark.asyncio
    async def test_returns_the_matching_record(self) -> None:
        """A known id resolves to its record."""
        result = await directory_get(_build_ctx(), "d-2")

        assert isinstance(result, DirectorySummary)
        assert result.id == "d-2"
        assert result.name == "Grace Hopper"

    @pytest.mark.asyncio
    async def test_returns_none_for_an_unknown_id(self) -> None:
        """An unknown id resolves to `None`, not an exception."""
        result = await directory_get(_build_ctx(), "d-does-not-exist")

        assert result is None

    @pytest.mark.asyncio
    async def test_identifier_whitespace_is_stripped(self) -> None:
        """Leading/trailing whitespace on `identifier` is tolerated (Convention 4)."""
        result = await directory_get(_build_ctx(), "  d-2  ")

        assert result is not None
        assert result.id == "d-2"
