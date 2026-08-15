"""Unit tests for the pure history trimmer (Req 3.1, 3.2).

Drives task 4.1's implementation (`app/stores/session_store/_trim.py`).
Task 4.1 seeded the subset needed to prove the algorithm correct while
building it: under-cap no-op, pairing preservation, retry-prompt closure,
head retention, and empty-parts dropping. Task 4.2 widens this file with the
remaining invariants from plan.md's "Trimming invariants (Req 3.2)" table
that 4.1 did not yet assert directly: forward-search termination (proving the
scan always finds a valid cut, including the fully-empty tail), idempotence,
order preservation, and the degenerate cap case.
"""

import pytest
from pydantic_ai.messages import ModelMessage
from pydantic_ai.messages import ModelRequest
from pydantic_ai.messages import ModelRequestPart
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import ModelResponsePart
from pydantic_ai.messages import RetryPromptPart
from pydantic_ai.messages import SystemPromptPart
from pydantic_ai.messages import TextPart
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.messages import UserPromptPart

from app.stores.session_store._trim import trim_history


def _req(*parts: ModelRequestPart) -> ModelRequest:
    """Build a ModelRequest from the given parts."""
    return ModelRequest(parts=list(parts))


def _resp(*parts: ModelResponsePart) -> ModelResponse:
    """Build a ModelResponse from the given parts."""
    return ModelResponse(parts=list(parts))


class TestUnderCap:
    """trim_history() is a no-op when the history is already within the cap."""

    def test_returns_unchanged_when_under_cap(self) -> None:
        """A history shorter than the cap is returned unchanged."""
        messages: list[ModelMessage] = [
            _req(SystemPromptPart(content="sys")),
            _req(UserPromptPart(content="hi")),
            _resp(TextPart(content="hello")),
        ]
        result = trim_history(messages, max_messages=10)
        assert result == messages

    def test_returns_unchanged_when_exactly_at_cap(self) -> None:
        """A history exactly at the cap is returned unchanged."""
        messages: list[ModelMessage] = [
            _req(SystemPromptPart(content="sys")),
            _req(UserPromptPart(content="hi")),
        ]
        result = trim_history(messages, max_messages=2)
        assert result == messages


class TestTrimWithoutToolCalls:
    """With no tool-call pairing at stake, the cut lands exactly on the cap."""

    def test_trims_to_exact_cap_when_no_pairing_constraint(self) -> None:
        """No tool parts anywhere means the ideal cut is always valid."""
        head = _req(SystemPromptPart(content="sys"))
        messages: list[ModelMessage] = [
            head,
            _req(UserPromptPart(content="q1")),
            _resp(TextPart(content="a1")),
            _req(UserPromptPart(content="q2")),
            _resp(TextPart(content="a2")),
        ]
        result = trim_history(messages, max_messages=3)
        assert len(result) == 3
        assert result[0] is head
        assert result[1:] == messages[3:]


class TestPairingPreservation:
    """The cut point never orphans a retained tool return."""

    def test_forward_search_drops_extra_message_to_avoid_orphan_return(self) -> None:
        """A cut that would split a call/return pair is pushed forward instead."""
        head = _req(SystemPromptPart(content="sys"))
        call = ToolCallPart(tool_name="lookup", args={}, tool_call_id="c1")
        ret = ToolReturnPart(tool_name="lookup", content="result", tool_call_id="c1")
        messages: list[ModelMessage] = [
            head,
            _req(UserPromptPart(content="q1")),
            _resp(call),
            _req(ret),
            _resp(TextPart(content="a1")),
            _req(UserPromptPart(content="q2")),
            _resp(TextPart(content="a2")),
        ]
        # Ideal boundary (max_messages=5) would cut right between the call and
        # its return, orphaning the return. The trimmer must search forward
        # and drop one more message rather than split the pair.
        result = trim_history(messages, max_messages=5)
        assert len(result) == 4
        assert result[0] is head
        assert result[1:] == messages[4:]
        # No retained ToolReturnPart lacks its originating ToolCallPart.
        opener_ids = {
            p.tool_call_id
            for m in result
            if isinstance(m, ModelResponse)
            for p in m.parts
            if isinstance(p, ToolCallPart)
        }
        closer_ids = {
            p.tool_call_id
            for m in result
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, ToolReturnPart)
        }
        assert closer_ids <= opener_ids

    def test_retry_prompt_naming_a_tool_closes_the_pair(self) -> None:
        """A RetryPromptPart naming a tool closes the pair like a ToolReturnPart."""
        head = _req(SystemPromptPart(content="sys"))
        call = ToolCallPart(tool_name="lookup", args={}, tool_call_id="c1")
        retry = RetryPromptPart(content="try again", tool_name="lookup", tool_call_id="c1")
        messages: list[ModelMessage] = [
            head,
            _req(UserPromptPart(content="q1")),
            _resp(call),
            _req(retry),
            _resp(TextPart(content="a1")),
            _req(UserPromptPart(content="q2")),
            _resp(TextPart(content="a2")),
        ]
        result = trim_history(messages, max_messages=5)
        assert len(result) == 4
        assert result[0] is head
        assert result[1:] == messages[4:]


class TestHeadRetention:
    """messages[0] is always retained, even at a cap of one."""

    def test_head_survives_even_when_it_would_be_the_naive_cut_point(self) -> None:
        """The pinned head is kept even when the cap admits only one message."""
        head = _req(SystemPromptPart(content="sys"))
        messages: list[ModelMessage] = [
            head,
            _resp(TextPart(content="a1")),
            _req(UserPromptPart(content="q2")),
            _resp(TextPart(content="a2")),
        ]
        result = trim_history(messages, max_messages=1)
        assert result == [head]


class TestNoEmptyParts:
    """No message with zero parts is ever emitted."""

    def test_message_with_no_parts_is_dropped(self) -> None:
        """A message with an empty parts list is dropped, not emitted as-is."""
        head = _req(SystemPromptPart(content="sys"))
        empty = _req()
        tail = _resp(TextPart(content="a1"))
        messages: list[ModelMessage] = [head, empty, tail]
        result = trim_history(messages, max_messages=10)
        assert empty not in result
        assert result == [head, tail]


class TestForwardSearchTermination:
    """The forward search always finds a valid cut and always terminates.

    Advancing (dropping more, never less) shrinks the closer set
    monotonically, so a valid `k` always exists — at worst the empty tail,
    where both the opener and closer sets are empty. These tests prove the
    search actually reaches that point rather than assuming it, per ADR-4.
    """

    def test_search_advances_past_multiple_invalid_candidates_before_terminating(
        self,
    ) -> None:
        """Three orphaned returns in a row each force one more step forward."""
        head = _req(SystemPromptPart(content="sys"))
        tail = _resp(TextPart(content="a1"))
        messages: list[ModelMessage] = [
            head,
            _req(UserPromptPart(content="q0")),
            _req(ToolReturnPart(tool_name="t1", content="r1", tool_call_id="orphan1")),
            _req(ToolReturnPart(tool_name="t2", content="r2", tool_call_id="orphan2")),
            _req(ToolReturnPart(tool_name="t3", content="r3", tool_call_id="orphan3")),
            tail,
        ]
        # ideal_cut = len(messages) - max_messages = 6 - 5 = 1, landing right
        # on the first orphaned return. None of orphan1/orphan2/orphan3 has a
        # matching call anywhere in the list, so each of the next three
        # candidates is invalid in turn until all three returns are dropped.
        result = trim_history(messages, max_messages=5)
        assert result == [head, tail]

    def test_search_terminates_at_a_full_drop_when_no_earlier_cut_is_valid(self) -> None:
        """When nothing before the empty tail is valid, the search still ends there."""
        head = _req(SystemPromptPart(content="sys"))
        messages: list[ModelMessage] = [
            head,
            _req(UserPromptPart(content="q1")),
            _resp(TextPart(content="a1")),
            _req(UserPromptPart(content="q2")),
            _resp(TextPart(content="a2")),
            _req(ToolReturnPart(tool_name="x", content="orphan", tool_call_id="never-called")),
        ]
        # The trailing return names a call that never occurred anywhere in
        # the list, so no candidate before k == len(rest) satisfies the
        # pairing rule. The search must still terminate at the empty tail
        # rather than raise or loop, dropping the return along with
        # everything else after the pinned head.
        result = trim_history(messages, max_messages=3)
        assert result == [head]


class TestIdempotence:
    """Re-trimming an already-trimmed history changes nothing (ADR-4)."""

    def test_trimming_a_trimmed_history_is_a_no_op(self) -> None:
        """A second pass at the same cap returns the first pass's result unchanged."""
        head = _req(SystemPromptPart(content="sys"))
        call = ToolCallPart(tool_name="lookup", args={}, tool_call_id="c1")
        ret = ToolReturnPart(tool_name="lookup", content="result", tool_call_id="c1")
        messages: list[ModelMessage] = [
            head,
            _req(UserPromptPart(content="q1")),
            _resp(call),
            _req(ret),
            _resp(TextPart(content="a1")),
            _req(UserPromptPart(content="q2")),
            _resp(TextPart(content="a2")),
        ]
        once = trim_history(messages, max_messages=5)
        twice = trim_history(once, max_messages=5)
        assert twice == once

    def test_trimming_an_already_under_cap_history_is_a_no_op(self) -> None:
        """Idempotence holds trivially in the under-cap branch too."""
        head = _req(SystemPromptPart(content="sys"))
        messages: list[ModelMessage] = [head, _req(UserPromptPart(content="q1"))]
        once = trim_history(messages, max_messages=10)
        twice = trim_history(once, max_messages=10)
        assert twice == once == messages


class TestOrderPreservation:
    """Trimming never reorders retained messages relative to the source list."""

    def test_retained_messages_keep_their_original_relative_order(self) -> None:
        """Each retained message's source position is strictly increasing."""
        head = _req(SystemPromptPart(content="sys"))
        call = ToolCallPart(tool_name="lookup", args={}, tool_call_id="c1")
        ret = ToolReturnPart(tool_name="lookup", content="result", tool_call_id="c1")
        messages: list[ModelMessage] = [
            head,
            _req(UserPromptPart(content="q1")),
            _resp(call),
            _req(ret),
            _req(UserPromptPart(content="q2")),
            _resp(TextPart(content="a2")),
            _req(UserPromptPart(content="q3")),
            _resp(TextPart(content="a3")),
        ]
        result = trim_history(messages, max_messages=5)
        # Map each retained message back to its source position by identity
        # (not equality, since structurally-equal messages must not collide);
        # those positions must be strictly increasing, proving no reordering.
        positions = [next(i for i, m in enumerate(messages) if m is r) for r in result]
        assert positions == sorted(positions)
        assert len(set(positions)) == len(positions)
        assert positions[0] == 0


class TestDegenerateCap:
    """At `max_messages <= 1` the head pin outranks the cap itself (ADR-4)."""

    @pytest.mark.parametrize("max_messages", [0, 1])
    def test_drops_entire_tail_regardless_of_pairing(self, max_messages: int) -> None:
        """Even an already-valid tail is dropped entirely once the cap admits only the head."""
        head = _req(SystemPromptPart(content="sys"))
        call = ToolCallPart(tool_name="lookup", args={}, tool_call_id="c1")
        ret = ToolReturnPart(tool_name="lookup", content="result", tool_call_id="c1")
        messages: list[ModelMessage] = [
            head,
            _resp(call),
            _req(ret),
            _resp(TextPart(content="a1")),
        ]
        # rest[0:] already satisfies the pairing rule on its own (the call
        # and its return are both present), so a naive forward search could
        # stop at k=0 and keep everything. The degenerate cap must still
        # drop the whole tail regardless of pairing validity.
        result = trim_history(messages, max_messages=max_messages)
        assert result == [head]
