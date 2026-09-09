"""Reference implementation of the real-tool design conventions (Req 15).

`docs/tool-design-conventions.md` states four conventions every real
(non-mock) agent tool must follow, each with an isolated code snippet. This
module is the missing piece: one small, working, lint/type-checked,
test-covered tool pair that applies all four **together**, so the first real
tool has a template to copy rather than four separate fragments to
reassemble from scratch.

`directory_search`/`directory_get` are deliberately **not** decorated with
`@agent.tool` and are never registered on `chat_agent` — registering them
would trip `real-tool-conventions-guard` (`.pre-commit-config.yaml`), whose
whole point is to force a conscious review the first time a *real* tool is
wired in. Copy the shape here, add the decorator, and go through that review
at that point.

The four conventions, applied:

1. **Naming** (`docs/tool-design-conventions.md#1`) — `directory_search` /
   `directory_get` share the `directory_` resource prefix.
2. **Pagination** (`#2`) — `directory_search` returns `next_offset`, `None`
   on the last page.
3. **`response_format`** (`#3`) — both tools accept `concise` (default) vs
   `detailed`.
4. **Lenient parsing** (`#4`) — `_normalize_response_format` tolerates case
   variance and whitespace; `_coerce_int` tolerates a numeric-looking string
   (a model occasionally emits `"5"` for an `int` parameter) while still
   rejecting genuinely malformed input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final
from typing import Literal

from pydantic import BaseModel
from pydantic_ai import RunContext

from app.agents.deps import AgentDeps


ResponseFormat = Literal["concise", "detailed"]

_DEFAULT_LIMIT: Final = 5
"""Page size when the caller omits `limit` — small by default (token efficiency)."""
_MAX_LIMIT: Final = 25
"""Hard ceiling on page size so one call can never flood the context window."""
_DETAIL_NOTE_CHARS: Final = 80
"""Truncation cap applied to a record's free-text notes in `detailed` output."""


@dataclass(frozen=True, slots=True)
class DirectoryRecord:
    """One directory entry the reference tools page, filter, and render over."""

    identifier: str
    name: str
    role: str
    notes: str


_RECORDS: Final[tuple[DirectoryRecord, ...]] = (
    DirectoryRecord("d-1", "Ada Lovelace", "Mathematician", "Wrote the first algorithm."),
    DirectoryRecord("d-2", "Grace Hopper", "Rear Admiral", "Popularized the term 'debugging'."),
    DirectoryRecord(
        "d-3", "Alan Turing", "Mathematician", "Formalized computation and computability."
    ),
)
"""Fixture data. A real tool wires this to a store/client via `ctx.deps` instead."""


class DirectorySummary(BaseModel):
    """Concise view of a `DirectoryRecord`: identity only."""

    id: str
    name: str


class DirectoryDetail(BaseModel):
    """Detailed view of a `DirectoryRecord`: identity plus truncated free text."""

    id: str
    name: str
    role: str
    notes: str


class DirectorySearchResult(BaseModel):
    """Response for `directory_search`."""

    total: int
    items: list[DirectorySummary | DirectoryDetail]
    next_offset: int | None
    """Pass this back as `offset` to fetch the next page. `None` on the last page."""


def _normalize_response_format(value: object) -> ResponseFormat:
    """Read the `response_format` knob leniently, defaulting to `concise`.

    Tolerates case and whitespace variance (`" Detailed "`, `"DETAILED"`);
    anything else — including a missing value — reads as the token-efficient
    default rather than raising.
    """
    if isinstance(value, str) and value.strip().lower() == "detailed":
        return "detailed"
    return "concise"


def _coerce_int(value: object, *, default: int, minimum: int, maximum: int | None = None) -> int:
    """Coerce `value` to a bounded int, tolerating a numeric-looking string.

    Falls back to `default` for anything unparsable or below `minimum`
    (including a `bool`, since `isinstance(True, int)` is `True` in Python)
    rather than raising — a stray model-emitted `"5"` for an `int` parameter
    is cosmetic noise, not a malformed call. Clamps down to `maximum` when
    given.
    """
    if isinstance(value, bool):
        return default
    if isinstance(value, str):
        try:
            value = int(value.strip())
        except ValueError:
            return default
    if not isinstance(value, int) or value < minimum:
        return default
    return min(value, maximum) if maximum is not None else value


def _truncate(text: str, limit: int) -> str:
    """Truncate `text` to `limit` chars, appending an ellipsis when shortened."""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _render(
    record: DirectoryRecord, response_format: ResponseFormat
) -> DirectorySummary | DirectoryDetail:
    """Render `record` at the requested verbosity (the token-efficiency knob)."""
    if response_format == "concise":
        return DirectorySummary(id=record.identifier, name=record.name)
    return DirectoryDetail(
        id=record.identifier,
        name=record.name,
        role=record.role,
        notes=_truncate(record.notes, _DETAIL_NOTE_CHARS),
    )


async def directory_search(
    ctx: RunContext[AgentDeps],
    query: str = "",
    offset: object = 0,
    limit: object = _DEFAULT_LIMIT,
    response_format: object = "concise",
) -> DirectorySearchResult:
    """Search the directory, paginated and filtered (reference tool — not registered).

    Args:
        ctx: RunContext providing access to AgentDeps. Unused here since the
            reference data is a module-level fixture; a real tool would read
            a store/client off `ctx.deps` instead.
        query: Case-insensitive substring filter over name/role/notes.
        offset: Zero-based page start. Accepts a numeric string leniently.
        limit: Page size, clamped to `_MAX_LIMIT`. Accepts a numeric string
            leniently.
        response_format: `"concise"` (id + name) or `"detailed"` (full
            record, notes truncated). Case/whitespace-tolerant.

    Returns:
        The matched `total`, the page `items`, and a `next_offset` cursor
        (`None` on the last page) so the caller can page forward only when
        it actually needs to.
    """
    del ctx
    normalized_query = query.strip().lower()
    resolved_limit = _coerce_int(limit, default=_DEFAULT_LIMIT, minimum=1, maximum=_MAX_LIMIT)
    resolved_offset = _coerce_int(offset, default=0, minimum=0)
    resolved_format = _normalize_response_format(response_format)

    matched = [
        record
        for record in _RECORDS
        if normalized_query in f"{record.name} {record.role} {record.notes}".lower()
    ]
    page = matched[resolved_offset : resolved_offset + resolved_limit]
    end = resolved_offset + len(page)
    next_offset = end if end < len(matched) else None
    return DirectorySearchResult(
        total=len(matched),
        items=[_render(record, resolved_format) for record in page],
        next_offset=next_offset,
    )


async def directory_get(
    ctx: RunContext[AgentDeps],
    identifier: str,
    response_format: object = "concise",
) -> DirectorySummary | DirectoryDetail | None:
    """Fetch one directory record by id (reference tool — not registered).

    The targeted counterpart to `directory_search`: skips straight to a known
    record instead of paging through the whole directory.

    Args:
        ctx: RunContext providing access to AgentDeps. Unused here — see
            `directory_search`.
        identifier: The record id to fetch.
        response_format: `"concise"` or `"detailed"`. Case/whitespace-tolerant.

    Returns:
        The matching record at the requested verbosity, or `None` if no
        record has that id.
    """
    del ctx
    resolved_format = _normalize_response_format(response_format)
    match = next((record for record in _RECORDS if record.identifier == identifier.strip()), None)
    if match is None:
        return None
    return _render(match, resolved_format)
