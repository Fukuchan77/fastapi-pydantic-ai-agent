"""Contract-drift guard: README's normative fences must not go stale (Req 14.1).

Scope decision: this guard only ever checks that something the README *shows*
is still real — never that the README shows *everything* real. The README is
allowed to omit newer members (e.g. it never demonstrates a `tool_called` or
`error` SSE event, nor `VectorStore.query_with_scores()`); it must never show
a class, field, method, or discriminator value that no longer exists. That
asymmetry is what keeps this guard passable without editing README.md, which
is outside this test file's boundary.

Checked fences (chosen because Tasks 5/9 in this spec already updated them in
lockstep with their runtime contracts, per research.md's ADR mitigation for
the "contract-drift chicken-and-egg" risk):

- The SSE "Response Stream" fence vs. `app.patterns.sse.SSEEvent` (Literal
  vocabulary + field set).
- The `VectorStore`/`SessionStore` Protocol extension-point example classes
  vs. the real Protocols (class set + method set).
- The chat/RAG JSON response examples vs. the real Pydantic response models
  (field set).

Not checked: the `/health` example (`{"status": "healthy", "service": ...}`)
already diverges from the real `{"status": "ok"}` response today, predating
this entire spec (no task touched `/health`, only `/health/ready`) - flagged
here as a candidate follow-up rather than silently ignored, mirroring Task 1's
precedent for documenting pre-existing gaps that are outside a task's own
boundary.
"""

import ast
import json
import re
import typing
from pathlib import Path

from app.models.agent import ChatResponse
from app.models.rag import IngestResponse
from app.models.rag import RAGQueryResponse
from app.patterns.sse import SSEEvent
from app.stores import session_store as session_store_module
from app.stores import vector_store as vector_store_module


_README = Path(__file__).resolve().parents[2] / "README.md"


def _readme_text() -> str:
    """Return the README's raw source text."""
    return _README.read_text(encoding="utf-8")


def _fence_after(text: str, marker: str, language: str = "") -> str:
    """Return the body of the first fenced code block appearing after `marker`."""
    marker_idx = text.index(marker)
    open_fence = f"```{language}"
    fence_start = text.index(open_fence, marker_idx)
    body_start = fence_start + len(open_fence)
    fence_end = text.index("```", body_start)
    return text[body_start:fence_end].strip("\n")


def _sse_event_classes() -> tuple[type, ...]:
    """Return the concrete event classes in the `SSEEvent` discriminated union."""
    union_type, *_annotations = typing.get_args(SSEEvent)
    return typing.get_args(union_type)


def _real_sse_types_and_fields() -> dict[str, set[str]]:
    """Map each real `type` discriminator value to its model's field names."""
    return {
        cls.model_fields["type"].default: set(cls.model_fields.keys())
        for cls in _sse_event_classes()
    }


class TestSSEContractDrift:
    """README's SSE "Response Stream" fence vs. `app.patterns.sse.SSEEvent`."""

    def _readme_events(self) -> list[dict[str, object]]:
        """Parse every `data: {...}` payload out of the SSE example fence via ast."""
        fence = _fence_after(_readme_text(), "**Response Stream:**")
        payloads = re.findall(r"^data: (\{.*\})$", fence, re.MULTILINE)
        return [ast.literal_eval(payload) for payload in payloads]

    def test_readme_demonstrates_at_least_one_event(self) -> None:
        """Guards against the fence/regex silently matching nothing."""
        assert self._readme_events()

    def test_every_readme_event_type_is_a_real_discriminator_value(self) -> None:
        """Every `type` value shown in the README must still be a real SSEEvent member."""
        real = _real_sse_types_and_fields()
        readme_types = {event["type"] for event in self._readme_events()}
        unknown = readme_types - real.keys()
        assert not unknown, f"README documents unknown SSE event type(s): {unknown}"

    def test_every_readme_event_field_set_matches_the_real_model(self) -> None:
        """Every example payload's field set must equal its real model's field set."""
        real = _real_sse_types_and_fields()
        for event in self._readme_events():
            event_type = str(event["type"])
            assert set(event.keys()) == real[event_type], (
                f"README's {event_type!r} example fields {set(event.keys())} "
                f"diverge from the real model's fields {real[event_type]}"
            )


class TestExtensionPointExampleDrift:
    """README's `VectorStore`/`SessionStore` Protocol extension-point examples."""

    def _class_and_methods(self, source: str) -> tuple[str, set[str]]:
        """Extract the sole class's name and method names from a fenced example."""
        tree = ast.parse(source)
        (class_def,) = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        methods = {
            node.name
            for node in class_def.body
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        return class_def.name, methods

    def test_custom_vector_store_example_matches_the_real_protocol(self) -> None:
        """The example class name and methods must still be real on `VectorStore`."""
        source = _fence_after(_readme_text(), "Implement [`VectorStore`]", "python")
        name, methods = self._class_and_methods(source)
        real_classes = {n for n, o in vars(vector_store_module).items() if isinstance(o, type)}
        assert name in real_classes, f"README's example class {name!r} no longer exists"
        real_methods = {m for m in dir(vector_store_module.VectorStore) if not m.startswith("_")}
        unknown = methods - real_methods
        assert not unknown, f"README's {name} example has stale method(s): {unknown}"

    def test_custom_session_store_example_matches_the_real_protocol(self) -> None:
        """The example class name and methods must still be real on `SessionStore`."""
        source = _fence_after(_readme_text(), "Implement [`SessionStore`]", "python")
        name, methods = self._class_and_methods(source)
        real_classes = {n for n, o in vars(session_store_module).items() if isinstance(o, type)}
        assert name in real_classes, f"README's example class {name!r} no longer exists"
        real_methods = {m for m in dir(session_store_module.SessionStore) if not m.startswith("_")}
        unknown = methods - real_methods
        assert not unknown, f"README's {name} example has stale method(s): {unknown}"


class TestJSONExampleFieldDrift:
    """README's chat/RAG JSON response examples vs. the real response models."""

    def _response_fields(self, marker: str) -> set[str]:
        """Parse the first `json` fence after `marker` and return its top-level keys."""
        fence = _fence_after(_readme_text(), marker, "json")
        payload = json.loads(fence)
        return set(payload.keys())

    def test_chat_response_example_fields_are_real(self) -> None:
        """The chat response example's fields must still be real on `ChatResponse`."""
        fields = self._response_fields('"message": "What is the weather like today?"')
        unknown = fields - set(ChatResponse.model_fields.keys())
        assert not unknown, f"README's chat response example has stale field(s): {unknown}"

    def test_ingest_response_example_fields_are_real(self) -> None:
        """The ingest response example's fields must still be real on `IngestResponse`."""
        fields = self._response_fields("FastAPI is a modern web framework")
        unknown = fields - set(IngestResponse.model_fields.keys())
        assert not unknown, f"README's ingest response example has stale field(s): {unknown}"

    def test_rag_query_response_example_fields_are_real(self) -> None:
        """The RAG query response example's fields must still be real on `RAGQueryResponse`."""
        fields = self._response_fields('"query": "What is Pydantic AI used for?"')
        unknown = fields - set(RAGQueryResponse.model_fields.keys())
        assert not unknown, f"README's RAG query response example has stale field(s): {unknown}"
