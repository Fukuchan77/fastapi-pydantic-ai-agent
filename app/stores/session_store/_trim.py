"""Pure history trimmer shared by every SessionStore implementation.

Cuts only at message boundaries and only where doing so cannot orphan a
retained tool return (Req 3.1, 3.2). See ADR-4 in
`.sdd/specs/002-review-roadmap-remediation/research.md` for the full
derivation of the validity rule and the forward-search direction.
"""

from collections.abc import Sequence

from pydantic_ai.messages import BaseToolCallPart
from pydantic_ai.messages import BaseToolReturnPart
from pydantic_ai.messages import ModelMessage
from pydantic_ai.messages import ModelRequest
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import RetryPromptPart


def trim_history(messages: Sequence[ModelMessage], max_messages: int) -> list[ModelMessage]:
    """Return at most `max_messages` of the most recent messages.

    `messages[0]` (the persisted system prompt) is always retained. The
    remaining cut point is searched forward from the ideal boundary until
    every retained tool return's originating call is also retained —
    advancing (dropping more, never less) guarantees a valid cut always
    exists, since the empty tail trivially satisfies the pairing rule.

    Args:
        messages: Full message history, in chronological order.
        max_messages: Maximum number of messages to retain.

    Returns:
        The trimmed history, at most `max_messages` long, in original order,
        with no message left holding zero parts.
    """
    if len(messages) <= max_messages:
        return [m for m in messages if m.parts]

    head, *rest = messages
    ideal_cut = len(messages) - max_messages
    cut = _find_valid_cut(rest, ideal_cut)
    trimmed = [head, *rest[cut:]]
    return [m for m in trimmed if m.parts]


def _find_valid_cut(rest: Sequence[ModelMessage], ideal_cut: int) -> int:
    """Find the smallest cut index >= ideal_cut where no closer is orphaned."""
    n = len(rest)
    start = min(max(ideal_cut, 0), n)
    for k in range(start, n + 1):
        candidates = rest[k:]
        if _closer_ids(candidates) <= _opener_ids(candidates):
            return k
    return n  # unreachable: k == n always satisfies the check (both sets empty)


def _opener_ids(messages: Sequence[ModelMessage]) -> set[str]:
    return {
        part.tool_call_id
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, BaseToolCallPart)
    }


def _closer_ids(messages: Sequence[ModelMessage]) -> set[str]:
    ids: set[str] = set()
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            is_closer = isinstance(part, BaseToolReturnPart) or (
                isinstance(part, RetryPromptPart) and part.tool_name is not None
            )
            if is_closer:
                ids.add(part.tool_call_id)
    return ids
