r"""Unit tests for the typed SSE event contract (app/patterns/sse.py)."""

import pytest
from pydantic import ValidationError

from app.patterns.sse import Completed
from app.patterns.sse import Error
from app.patterns.sse import StepStarted
from app.patterns.sse import Token
from app.patterns.sse import ToolCalled
from app.patterns.sse import parse_sse_events
from app.patterns.sse import to_sse


class TestToSSE:
    r"""to_sse() must frame each event as `event: <type>\ndata: <json>\n\n`."""

    def test_step_started_framing(self) -> None:
        """StepStarted serializes with no extra fields beyond the discriminator."""
        wire = to_sse(StepStarted())
        assert wire == 'event: step_started\ndata: {"type":"step_started"}\n\n'

    def test_token_framing_and_payload(self) -> None:
        """Token frames as event: token with the content field in the JSON payload."""
        wire = to_sse(Token(content="Hello"))
        assert wire.startswith("event: token\ndata: ")
        assert wire.endswith("\n\n")
        assert '"content":"Hello"' in wire

    def test_tool_called_framing_and_payload(self) -> None:
        """ToolCalled frames as event: tool_called with name and args_summary."""
        wire = to_sse(ToolCalled(name="mock_web_search", args_summary='{"query":"x"}'))
        assert wire.startswith("event: tool_called\ndata: ")
        assert '"name":"mock_web_search"' in wire
        assert '"args_summary"' in wire

    def test_completed_framing(self) -> None:
        """Completed serializes with no extra fields beyond the discriminator."""
        wire = to_sse(Completed())
        assert wire == 'event: completed\ndata: {"type":"completed"}\n\n'

    def test_error_framing_and_payload(self) -> None:
        """Error frames as event: error with the message field in the JSON payload."""
        wire = to_sse(Error(message="An unexpected error occurred"))
        assert wire.startswith("event: error\ndata: ")
        assert '"message":"An unexpected error occurred"' in wire

    def test_event_name_equals_type_discriminator(self) -> None:
        """The SSE event: name always equals the JSON type discriminator (Req 2.1)."""
        for event in (
            StepStarted(),
            ToolCalled(name="t", args_summary="s"),
            Token(content="x"),
            Completed(),
            Error(message="m"),
        ):
            wire = to_sse(event)
            event_line = wire.split("\n", 1)[0]
            assert event_line == f"event: {event.type}"


class TestErrorPayloadSafety:
    """Error events must never carry raw exception internals (Req 2.2)."""

    def test_error_model_has_no_traceback_field(self) -> None:
        """Error only exposes type and message, never traceback/exception internals."""
        assert set(Error.model_fields) == {"type", "message"}

    def test_error_message_is_a_plain_string(self) -> None:
        """Error.message round-trips as the plain string it was constructed with."""
        error = Error(message="An unexpected error occurred")
        assert error.message == "An unexpected error occurred"


class TestParseSSEEvents:
    """parse_sse_events() round-trips to_sse() output and is line-terminator-safe."""

    def test_round_trip_single_event(self) -> None:
        """A single to_sse() frame parses back into the original event."""
        original = Token(content="hello")
        parsed = parse_sse_events(to_sse(original))
        assert parsed == [original]

    def test_round_trip_multiple_events(self) -> None:
        """Concatenated frames for several events all parse back in order."""
        events = [
            StepStarted(),
            Token(content="Hel"),
            Token(content="lo"),
            Completed(),
        ]
        raw = "".join(to_sse(e) for e in events)
        assert parse_sse_events(raw) == events

    def test_round_trip_preserves_discriminated_types(self) -> None:
        """Parsed events keep their concrete Python type, not just equal field values."""
        events = [ToolCalled(name="search", args_summary="q=python"), Error(message="boom")]
        raw = "".join(to_sse(e) for e in events)
        parsed = parse_sse_events(raw)
        assert isinstance(parsed[0], ToolCalled)
        assert isinstance(parsed[1], Error)

    def test_line_separator_inside_payload_does_not_break_framing(self) -> None:
        """U+2028 (LINE SEPARATOR) in payload text must survive round-trip (Req 2.9)."""
        separator = chr(0x2028)
        original = Token(content=f"line1{separator}line2")
        raw = to_sse(original)
        # Pydantic's model_dump_json does not escape U+2028; a naive
        # str.splitlines()-based parser would incorrectly split the data line here.
        assert separator in raw
        parsed = parse_sse_events(raw)
        assert parsed == [original]

    def test_paragraph_separator_inside_payload_does_not_break_framing(self) -> None:
        """U+2029 (PARAGRAPH SEPARATOR) in payload text must survive round-trip (Req 2.9)."""
        separator = chr(0x2029)
        original = Token(content=f"para1{separator}para2")
        raw = to_sse(original)
        assert separator in raw
        parsed = parse_sse_events(raw)
        assert parsed == [original]

    def test_splits_only_on_crlf_cr_lf(self) -> None:
        r"""Mixed \r\n, \r, and \n terminators between frames all parse correctly."""
        raw = (
            'event: token\r\ndata: {"type":"token","content":"a"}\r\n\r\n'
            'event: token\rdata: {"type":"token","content":"b"}\r\r'
            'event: token\ndata: {"type":"token","content":"c"}\n\n'
        )
        parsed = parse_sse_events(raw)
        assert parsed == [Token(content="a"), Token(content="b"), Token(content="c")]

    def test_empty_input_returns_no_events(self) -> None:
        """Parsing an empty string yields an empty list, not an error."""
        assert parse_sse_events("") == []


class TestPayloadFieldConstraints:
    """The typed union's field sets should not admit raw prompts/creds by construction."""

    def test_step_started_and_completed_carry_no_extra_fields(self) -> None:
        """Bare marker events expose only the type discriminator."""
        assert set(StepStarted.model_fields) == {"type"}
        assert set(Completed.model_fields) == {"type"}

    def test_tool_called_fields_are_name_and_summary_only(self) -> None:
        """ToolCalled exposes name/args_summary only, never raw tool call args."""
        assert set(ToolCalled.model_fields) == {"type", "name", "args_summary"}

    def test_token_fields_are_content_only(self) -> None:
        """Token exposes content only."""
        assert set(Token.model_fields) == {"type", "content"}

    def test_extra_fields_are_rejected(self) -> None:
        """Unknown fields (e.g. an accidentally-leaked raw_prompt) fail validation."""
        with pytest.raises(ValidationError):
            Token(content="x", raw_prompt="leaked")  # type: ignore[call-arg]
