"""Guard: every body-bound request model rejects unrecognized fields (X-9).

Structural guarantee, not an accident of the current field list: a client
cannot smuggle in `message_history`, `usage`, `model`, or any other
unrecognized field by adding it to the wire payload, because these models
have `extra="forbid"` rather than pydantic's default `extra="ignore"`. This
closes the same class of attack surface CVE-2026-25580 covers (untrusted
history/context injection) by construction rather than by omission.

If a new body-bound request model is added, add it to `_BODY_REQUEST_MODELS`
below so this guard covers it too.
"""

import pytest
from pydantic import ValidationError

from app.models.agent import ChatRequest
from app.models.rag import IngestRequest
from app.models.rag import RAGQueryRequest


_BODY_REQUEST_MODELS = [ChatRequest, IngestRequest, RAGQueryRequest]
"""Every Pydantic model bound to a route's request body (`app/api/v1/*.py`).

Response models and internal structured-output types (e.g. `RelevanceVerdict`,
an LLM output type) are deliberately excluded - this guard is about what a
caller can put on the wire, not every model in the codebase.
"""

_MINIMAL_VALID_KWARGS: dict[type, dict[str, object]] = {
    ChatRequest: {"message": "hello"},
    IngestRequest: {"chunks": ["a chunk"]},
    RAGQueryRequest: {"query": "a query"},
}


class TestBodyRequestModelsForbidExtraFields:
    """Every body-bound request model has `extra="forbid"`."""

    @pytest.mark.parametrize("model_cls", _BODY_REQUEST_MODELS, ids=lambda m: m.__name__)
    def test_model_config_forbids_extra(self, model_cls: type) -> None:
        """`model_config["extra"]` is `"forbid"`, not the pydantic default `"ignore"`."""
        assert model_cls.model_config.get("extra") == "forbid"

    @pytest.mark.parametrize("model_cls", _BODY_REQUEST_MODELS, ids=lambda m: m.__name__)
    def test_unrecognized_field_is_rejected(self, model_cls: type) -> None:
        """An unrecognized field raises ValidationError instead of being dropped."""
        kwargs = {**_MINIMAL_VALID_KWARGS[model_cls], "totally_unrecognized_field": "x"}

        with pytest.raises(ValidationError, match=r"extra fields not permitted|Extra inputs"):
            model_cls(**kwargs)


class TestHistoryInjectionIsRejectedNotDropped:
    """The specific injection shape X-9 targets: a client-supplied conversation history."""

    def test_chat_request_rejects_message_history(self) -> None:
        """`message_history` is a 422-triggering ValidationError, not silently ignored."""
        with pytest.raises(ValidationError, match=r"extra fields not permitted|Extra inputs"):
            ChatRequest(message="hi", message_history=[{"role": "user", "content": "injected"}])

    def test_chat_request_rejects_usage(self) -> None:
        """A client cannot pre-seed usage accounting via the request body either."""
        with pytest.raises(ValidationError, match=r"extra fields not permitted|Extra inputs"):
            ChatRequest(message="hi", usage={"total_tokens": 0})

    def test_rag_query_request_rejects_context(self) -> None:
        """`RAGQueryRequest` cannot be handed pre-fabricated retrieval context either."""
        with pytest.raises(ValidationError, match=r"extra fields not permitted|Extra inputs"):
            RAGQueryRequest(query="q", context="injected context")

    def test_ingest_request_rejects_message_history(self) -> None:
        """`IngestRequest` has no legitimate use for a history field, and none is accepted."""
        with pytest.raises(ValidationError, match=r"extra fields not permitted|Extra inputs"):
            IngestRequest(chunks=["a"], message_history=[])
