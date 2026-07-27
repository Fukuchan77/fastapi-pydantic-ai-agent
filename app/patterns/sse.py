r"""Typed 5-event SSE contract and a line-terminator-safe wire codec.

Wire format: each event is `event: <type>\ndata: <json>\n\n`, where `<type>`
is the event's `type` discriminator and `<json>` is its compact JSON
serialization. `parse_sse_events()` splits only on SSE line terminators
(`\r\n`, `\r`, `\n`) rather than `str.splitlines()`, so U+2028/U+2029
characters inside a JSON payload (which pydantic's `model_dump_json()` does
not escape) are never mistaken for frame boundaries.
"""

import re
from typing import Annotated
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import TypeAdapter


class _StrictEvent(BaseModel):
    """Shared base: rejects unknown fields so payloads can never grow raw/sensitive data."""

    model_config = ConfigDict(extra="forbid")


class StepStarted(_StrictEvent):
    """Emitted when the agent begins a new model-request step."""

    type: Literal["step_started"] = "step_started"


class ToolCalled(_StrictEvent):
    """Emitted when the agent invokes a tool.

    Attributes:
        name: The tool's name.
        args_summary: A truncated, non-sensitive summary of the call arguments
            (never the raw prompt or credentials).
    """

    type: Literal["tool_called"] = "tool_called"
    name: str
    args_summary: str


class Token(_StrictEvent):
    """Emitted for each text delta produced by the model.

    Attributes:
        content: The delta text content.
    """

    type: Literal["token"] = "token"
    content: str


class Completed(_StrictEvent):
    """Emitted once when the agent run reaches its final result."""

    type: Literal["completed"] = "completed"


class Error(_StrictEvent):
    """Emitted as the terminal event when the stream fails.

    Attributes:
        message: A generic, safe-to-expose error message (never a stack trace).
    """

    type: Literal["error"] = "error"
    message: str


SSEEvent = Annotated[
    StepStarted | ToolCalled | Token | Completed | Error,
    Field(discriminator="type"),
]

_sse_event_adapter: TypeAdapter[SSEEvent] = TypeAdapter(SSEEvent)

_LINE_SPLIT = re.compile(r"\r\n|\r|\n")
_DATA_PREFIX = "data:"
_EVENT_PREFIX = "event:"


def to_sse(event: SSEEvent) -> str:
    r"""Serialize a typed SSE event into `event: <type>\ndata: <json>\n\n` wire format.

    Args:
        event: The typed event to serialize.

    Returns:
        The framed SSE wire text for this event.
    """
    return f"{_EVENT_PREFIX} {event.type}\n{_DATA_PREFIX} {event.model_dump_json()}\n\n"


def parse_sse_events(raw: str) -> list[SSEEvent]:
    r"""Parse concatenated SSE wire text back into typed events.

    Splits only on SSE line terminators (`\r\n`, `\r`, `\n`) — never on the
    wider Unicode line-boundary set that `str.splitlines()` uses — so
    U+2028/U+2029 characters inside a JSON payload cannot be mistaken for
    frame boundaries.

    Args:
        raw: Concatenated SSE wire text (one or more framed events).

    Returns:
        The parsed events, in the order they appeared.
    """
    events: list[SSEEvent] = []
    data_lines: list[str] = []

    def _flush() -> None:
        if data_lines:
            events.append(_sse_event_adapter.validate_json("\n".join(data_lines)))
            data_lines.clear()

    for line in _LINE_SPLIT.split(raw):
        if line.startswith(_DATA_PREFIX):
            data_lines.append(line[len(_DATA_PREFIX) :].lstrip(" "))
        elif line.startswith(_EVENT_PREFIX):
            continue  # the discriminator is already embedded in the JSON `type` field
        elif line == "":
            _flush()
    _flush()
    return events
