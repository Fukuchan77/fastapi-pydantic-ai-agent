"""Regression tests for the generation-keyed RAG result cache (Req 2).

Covers 2.7 end-to-end: a query that returned a stale result before an ingest
must return post-ingest content on the next identical query without waiting
for the TTL (2.1), a failed ingest must leave the generation unchanged (2.4),
a query already in flight when an ingest advances the generation must still
resolve to its own result (2.3), and the cache key must never depend on
caller identity (2.6). Uses a real `InMemoryVectorStore` and
`CorrectiveRAGWorkflow` with a `FunctionModel`, per the integration-tier
convention in `test_rag_workflow.py` and `test_corrective_rag_timeout.py`.
"""

import asyncio
import contextlib

import pytest
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.models.function import FunctionModel

from app.security.principal import derive_principal_id
from app.stores.vector_store import InMemoryVectorStore
from app.workflows.corrective_rag import CorrectiveRAGWorkflow
from tests.conftest import build_test_settings


# Upper bound for every inter-task handshake in this module.
#
# These waits used to be unbounded. `eval_started` is set only when the eval
# agent is invoked with output tools, so anything that stops the workflow from
# reaching its structured-output call - a bug, or an earlier failure in the
# same module - leaves the event unset and `wait()` blocks forever. That is not
# hypothetical: PR CI run 31862477318 hung here for 5h59m and was killed by the
# GitHub Actions 6-hour job limit, which meant pytest never printed its failure
# summary and the tracebacks for the eight tests that had already failed were
# lost with it. A test must fail when the thing it waits for does not happen.
_EVENT_WAIT_TIMEOUT = 30.0


@pytest.mark.asyncio
async def test_stale_query_returns_post_ingest_content_without_ttl_wait() -> None:
    """2.1/2.4/2.5/2.7: an ingest invalidates cached results without a TTL wait.

    A query cached before an ingest must miss on the next identical query
    once the ingest succeeds (2.1, 2.7), without waiting for `rag_cache_ttl`.
    A *failed* ingest must leave the generation - and therefore the cached
    result - untouched (2.4), and a repeated identical query must still be
    served from cache while no successful ingest has occurred (2.5).
    """
    vector_store = InMemoryVectorStore()
    await vector_store.add_documents(["pre-ingest chunk about dogs"])

    # A long TTL so any post-ingest cache miss can only be explained by the
    # generation bump, never by the TTL expiring on its own.
    settings = build_test_settings(llm_model="openai:gpt-4", rag_cache_ttl=300)

    call_count = 0

    def model_fn(messages: list, info: AgentInfo) -> ModelResponse:
        """Accept every search result, then echo the synthesis prompt.

        The eval agent's structured tool call is always answered as
        sufficient so the workflow proceeds to synthesis without extra
        search rounds; the synthesis (plain-text) call is echoed back
        verbatim so the returned answer reveals exactly which chunk(s) it
        was grounded on. Discriminates on `info.output_tools` rather than
        call parity, since pydantic-ai's own output-retry loop can consume
        extra eval-agent calls that would otherwise shift the parity.
        """
        nonlocal call_count
        call_count += 1
        if info.output_tools:
            tool = info.output_tools[0]
            return ModelResponse(
                parts=[ToolCallPart(tool.name, {"sufficient": True, "rationale": "relevant"})]
            )
        prompt = messages[-1].parts[0].content if messages else ""
        return ModelResponse(parts=[TextPart(content=str(prompt))])

    workflow = CorrectiveRAGWorkflow(
        vector_store=vector_store,
        llm_settings=settings,
        llm_model=FunctionModel(model_fn),
    )

    stale_result = await workflow.run(query="dogs", max_retries=1)
    assert "pre-ingest chunk about dogs" in stale_result["answer"]

    # 2.4: a failed ingest (oversized chunk, rejected before any document is
    # added) must leave the generation - and thus the cache key - unchanged.
    generation_before_failed_ingest = vector_store.generation
    oversized_chunk = "x" * (vector_store.max_chunk_size + 1)
    with pytest.raises(ValueError, match="too large"):
        await vector_store.add_documents([oversized_chunk])
    assert vector_store.generation == generation_before_failed_ingest

    # 2.5: with no successful ingest yet, the identical query is still
    # served from cache - no new model calls, identical cached answer.
    calls_before_repeat = call_count
    still_cached_result = await workflow.run(query="dogs", max_retries=1)
    assert call_count == calls_before_repeat, "a cache hit must not invoke the LLM again"
    assert still_cached_result == stale_result

    # 2.1/2.7: a successful ingest advances the generation, so the next
    # identical query misses the stale entry and reflects the new content -
    # immediately, not after `rag_cache_ttl` seconds.
    await vector_store.add_documents(["post-ingest chunk about dogs"])
    fresh_result = await workflow.run(query="dogs", max_retries=1)
    assert "post-ingest chunk about dogs" in fresh_result["answer"]
    assert fresh_result["answer"] != stale_result["answer"]


@pytest.mark.asyncio
async def test_pending_query_survives_generation_bump_during_ingest() -> None:
    """2.3: an ingest mid-flight must not disturb an already-registered future.

    Starts a query, lets it register its in-flight future under the
    pre-ingest generation's cache key, advances the generation with an
    ingest while that query is still pending, and asserts the pending query
    still resolves to its own result rather than erroring or being
    cancelled.
    """
    vector_store = InMemoryVectorStore()
    await vector_store.add_documents(["pre-ingest chunk about cats"])
    settings = build_test_settings(llm_model="openai:gpt-4", rag_cache_ttl=300)

    eval_started = asyncio.Event()
    release_eval = asyncio.Event()
    call_count = 0

    async def model_fn(messages: list, info: AgentInfo) -> ModelResponse:
        """Block the evaluation call until the test signals the ingest is done."""
        nonlocal call_count
        call_count += 1
        if info.output_tools:
            eval_started.set()
            async with asyncio.timeout(_EVENT_WAIT_TIMEOUT):
                await release_eval.wait()
            tool = info.output_tools[0]
            return ModelResponse(
                parts=[ToolCallPart(tool.name, {"sufficient": True, "rationale": "relevant"})]
            )
        return ModelResponse(parts=[TextPart(content="answer for pending query")])

    workflow = CorrectiveRAGWorkflow(
        vector_store=vector_store,
        llm_settings=settings,
        llm_model=FunctionModel(model_fn),
    )

    cache_key_before_ingest = workflow._generate_cache_key("cats", 1)

    query_task = asyncio.create_task(workflow.run(query="cats", max_retries=1))
    try:
        async with asyncio.timeout(_EVENT_WAIT_TIMEOUT):
            await eval_started.wait()
    except TimeoutError:  # pragma: no cover - only on a regression
        # Surface why the eval call never happened instead of reporting a bare
        # timeout: if the workflow already failed, its exception is the real
        # diagnosis and awaiting the task re-raises it here.
        query_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await query_task
        raise

    # The leader has registered its pending future under the pre-ingest key.
    assert cache_key_before_ingest in workflow._pending_futures

    # Ingest while the query above is still in flight: advances the
    # generation, so any *new* query would compute a different key, but the
    # already-registered future must be left alone.
    await vector_store.add_documents(["post-ingest chunk about cats"])
    cache_key_after_ingest = workflow._generate_cache_key("cats", 1)
    assert cache_key_after_ingest != cache_key_before_ingest

    release_eval.set()
    result = await asyncio.wait_for(query_task, timeout=5)

    assert result["answer"] == "answer for pending query"
    assert result["context_found"] is True
    assert cache_key_before_ingest in workflow._cache
    assert cache_key_before_ingest not in workflow._pending_futures


@pytest.mark.asyncio
async def test_cache_key_is_identical_across_callers_with_different_session_identity() -> None:
    """2.6: the cache key must never depend on caller identity.

    Runs the same query/retry-limit pair through two workflow instances
    constructed with different session-scoped inputs - distinct caller API
    keys, from which distinct principal ids are derived exactly as
    `app/deps/auth.py` + `app/security/principal.py` would for two different
    requests. Those identifiers are held only here, at the surrounding
    "request" layer, and are never passed to the workflow constructor or to
    `run()`. Both instances must still produce the identical cache key/entry.
    """
    vector_store = InMemoryVectorStore()
    await vector_store.add_documents(["shared corpus chunk"])

    def model_fn(messages: list, info: AgentInfo) -> ModelResponse:
        """Always accept the search result so both runs reach the same cache path."""
        if info.output_tools:
            tool = info.output_tools[0]
            return ModelResponse(
                parts=[ToolCallPart(tool.name, {"sufficient": True, "rationale": "relevant"})]
            )
        return ModelResponse(parts=[TextPart(content="synthesized answer")])

    caller_a_api_key = "caller-a-api-key-1234567890"
    caller_b_api_key = "caller-b-api-key-1234567890"
    principal_a = derive_principal_id(caller_a_api_key)
    principal_b = derive_principal_id(caller_b_api_key)
    assert principal_a != principal_b, "the two callers must be genuinely distinct identities"

    settings_a = build_test_settings(api_key=caller_a_api_key, llm_model="openai:gpt-4")
    settings_b = build_test_settings(api_key=caller_b_api_key, llm_model="openai:gpt-4")

    workflow_a = CorrectiveRAGWorkflow(
        vector_store=vector_store,
        llm_settings=settings_a,
        llm_model=FunctionModel(model_fn),
    )
    workflow_b = CorrectiveRAGWorkflow(
        vector_store=vector_store,
        llm_settings=settings_b,
        llm_model=FunctionModel(model_fn),
    )

    key_a = workflow_a._generate_cache_key("shared query", 3)
    key_b = workflow_b._generate_cache_key("shared query", 3)
    assert key_a == key_b, "the cache key must not depend on caller identity"

    await workflow_a.run(query="shared query", max_retries=3)
    await workflow_b.run(query="shared query", max_retries=3)
    assert key_a in workflow_a._cache
    assert key_b in workflow_b._cache
