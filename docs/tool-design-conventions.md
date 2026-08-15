# Real Agent Tool Design Conventions

**Status: deferred.** No real (non-mock) agent tool exists in this codebase yet — the
only registered tool is [`mock_web_search`](../app/agents/tools_mock.py), a
development-only stub that returns canned data and is never registered in
`production` (see `app/agents/chat_agent.py`'s `register_mock_tools` gating).

This document defines the conventions that **every future real tool MUST follow**
before it is registered on the chat agent (Requirement 15). It exists now, ahead of
any real tool, so the conventions are settled and reviewable before the first real
tool's design, rather than retrofitted after the fact.

## Table of Contents

- [1. Naming: `<resource>_<verb>`](#1-naming-resource_verb)
- [2. Pagination: `next_offset`](#2-pagination-next_offset)
- [3. `response_format`: `concise` / `detailed`](#3-response_format-concise--detailed)
- [4. Lenient argument parsing](#4-lenient-argument-parsing)
- [Enforcement](#enforcement)

---

## 1. Naming: `<resource>_<verb>`

Every real tool function name SHALL be `<resource>_<verb>` in `snake_case` — the
resource first, the action second (Req 15.1).

```python
# Good
async def documents_search(ctx: RunContext[AgentDeps], query: str) -> str: ...
async def orders_list(ctx: RunContext[AgentDeps], status: str | None = None) -> str: ...
async def ticket_create(ctx: RunContext[AgentDeps], title: str, body: str) -> str: ...

# Bad — verb-first, or no resource at all
async def search_documents(...): ...
async def list(...): ...
async def do_the_thing(...): ...
```

Rationale: a consistent `<resource>_<verb>` prefix keeps a growing tool list
sorted and scannable by resource in the model's tool listing (`documents_search`,
`documents_fetch`, `documents_delete` group together), which is easier for the
model — and the developer — to disambiguate than a verb-first name.

## 2. Pagination: `next_offset`

Every real tool that returns a list of items SHALL paginate its results and
return a `next_offset` in its response (Req 15.2).

```python
class DocumentsSearchResult(BaseModel):
    """Response for documents_search."""

    items: list[DocumentSummary]
    next_offset: int | None
    """Pass this back as `offset` to fetch the next page. None if this is the last page."""


async def documents_search(
    ctx: RunContext[AgentDeps],
    query: str,
    offset: int = 0,
    limit: int = 20,
) -> DocumentsSearchResult: ...
```

Rationale: an unbounded list-returning tool is an unbounded-token-consumption risk
(the model can trigger arbitrarily large tool outputs) and a poor UX for the model,
which has to reason about huge result sets in one shot. `next_offset` gives the
model an explicit, mechanical way to ask for "the rest" only when it actually needs
to, one bounded page at a time. `None` unambiguously signals the last page — do not
overload `0` or an empty string for "no more results".

## 3. `response_format`: `concise` / `detailed`

Every real tool SHALL accept a `response_format: Literal["concise", "detailed"] =
"concise"` parameter (Req 15.3).

```python
async def documents_fetch(
    ctx: RunContext[AgentDeps],
    document_id: str,
    response_format: Literal["concise", "detailed"] = "concise",
) -> str:
    document = await ctx.deps.doc_store.get(document_id)
    if response_format == "detailed":
        return document.full_text_with_metadata()
    return document.summary()
```

Rationale: most tool calls only need a short answer to keep reasoning on track;
occasionally the model needs the full payload to quote or analyze in detail.
Defaulting to `"concise"` keeps typical tool-call token cost — and therefore
context-window pressure and latency — low, while `"detailed"` remains available
on demand rather than force-fitting every caller into one fixed verbosity.

## 4. Lenient argument parsing

Every real tool SHALL parse its arguments leniently, tolerating minor formatting
variance from the model rather than raising a validation error (Req 15.4).

Minor variance that a real tool MUST tolerate:

- Case-insensitive matching for enum-like string arguments (`"Concise"`,
  `"CONCISE"`, `"concise"` are all `"concise"`).
- Leading/trailing whitespace on string arguments.
- Numeric-looking strings passed where an `int`/`float` is expected (`"20"` for
  `limit: int`), since models occasionally emit numbers as strings.
- A single item passed where a list of one is expected, and vice versa in the
  reasonable case.

```python
def _normalize_response_format(value: str) -> Literal["concise", "detailed"]:
    normalized = value.strip().lower()
    if normalized not in {"concise", "detailed"}:
        raise ValueError(f"response_format must be 'concise' or 'detailed', got {value!r}")
    return normalized
```

Rationale: an LLM-generated tool call is not a trusted, machine-validated API
client — treating a stray capital letter or trailing space as a hard failure burns
a retry (and the associated latency/token cost) on something a human reviewer
would consider obviously fine. This is *not* license to skip validation entirely:
still reject genuinely malformed input (e.g. an unrecognized `response_format`
value after normalization) — just don't fail on cosmetic variance.

---

## Enforcement

A pre-commit hook, `real-tool-conventions-guard`
(`.pre-commit-config.yaml`), is a **guard stub**: it is inert today (there is
nothing for it to flag), and it activates automatically the moment a real
`@agent.tool`-decorated function is added anywhere under `app/agents/` outside
`app/agents/tools_mock.py`. When it fires, it blocks the commit until a human
reviews the new tool against the four conventions above and — if compliant —
extends the hook (or adds an equivalent runtime test) to encode the real check
mechanically, rather than silently letting the first real tool skip review.

This is deliberately a *reminder-and-block* stub, not a conventions checker: statically
verifying "is this function's argument parsing lenient?" or "does this list-returning
tool paginate correctly?" requires understanding the function's runtime behavior, not
just its shape — that verification belongs in code review and the new tool's own tests
at the point a real tool is actually introduced, informed by this document.
