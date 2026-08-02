# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`AGENTS.md` is a condensed sibling of this file for other agent tools — when you change a convention here, update it there too.

## Commands

All tooling runs through `mise` (which wraps `uv`). Check `mise.toml` before running bare tools.

```bash
mise run dev                 # uvicorn dev server, hot reload, :8000
mise run test                # full suite with coverage
mise run test:unit           # fast, real network blocked
mise run test:integration    # real stores + FunctionModel LLM
mise run test:e2e            # full HTTP stack (AsyncClient)
mise run test:ci             # what PR CI runs: unit+integration+e2e + coverage gate
mise run test:benchmark      # latency/throughput/cache-hit benchmarks (-s)
mise run test:local          # requires a running Ollama instance (-m ollama)
mise run evals               # offline LLM-judge golden set; makes REAL LLM calls
mise run lint                # ruff check + ty check (type checker is `ty`, NOT mypy)
mise run format              # ruff format
mise run audit               # pip-audit dependency vulnerability scan
mise run hooks:install       # install the pre-commit hook
mise run build               # docker build
```

Run a single test: `uv run pytest tests/unit/stores/test_session_store.py::test_name -v`

Lint/type config lives in `pyproject.toml`: Ruff is strict (`S` bandit, `ANN` annotations, `D` google-style docstrings, `SIM`, `B`), imports are one-per-line (`force-single-line`), line length 100, Python 3.13+. `ty` runs in strict mode (no implicit `Any`). Tests relax `S101`/`ANN`. Coverage gate is `fail_under = 80`. Pytest markers: `docker`, `benchmark`, `ollama`.

`mise run lint`/`format` cover `app/`, `evals/`, and `tests/` — `evals/` is production-linted code, not a scratch directory.

### Gating: what runs where

- **PR CI** (`.github/workflows/pr.yml`): lint → `test:ci` → `audit`. Never runs live LLM or Ollama tests.
- **Nightly** (`.github/workflows/security.yml`): `pip-audit` + gitleaks, cron `37 3 * * *`.
- **pre-commit** (`.pre-commit-config.yaml`): gitleaks, `pip-audit`, a pygrep `no-hardcoded-model-id` guard, and an inert `real-tool-conventions-guard` that fires the moment a non-mock `@agent.tool` appears under `app/agents/` (forcing a read of `docs/tool-design-conventions.md`).
- **pre-push** (`.githooks/pre-push`, enable with `git config core.hooksPath .githooks`): availability-gated — probes `${OLLAMA_BASE_URL}/api/tags`, runs `test:local` + `evals` when reachable, warns and lets the push through when not.

## Architecture

A FastAPI framework demonstrating three agent patterns. `README.md` describes the original design; the code has since grown guardrails, session ownership, provider fallback chains, Redis/Chroma/Ollama backends, and an evals harness — trust the code over the README where they differ (a guard test keeps the README's *shown* examples honest; see Repo guards below).

### Composition root

`app/main.py::create_app(settings=None, model=None)` is the factory. `app = create_app()` at the module bottom is only the `uvicorn app.main:app` entrypoint — **tests always call `create_app(settings=..., model=test_model)`** rather than importing the module-level singleton, which is what lets them avoid monkeypatching env vars.

`lifespan` populates `app.state` with the shared singletons everything else reads: `settings`, `http_client` (an `httpx.AsyncClient` over a custom `RetryTransport` that retries only transient 5xx `{500,502,503,504}` plus network errors), `vector_store`, `session_store`, `chat_agent`, and a background `cleanup_task`. Startup is **fail-fast**: stores are built from settings, probed via `dry_run_stores()`, and the `FallbackModel` chain is built eagerly — a misconfigured provider or unreachable Redis/Ollama kills startup instead of surfacing on the first request. A test-injected `model` bypasses the fallback build entirely.

**Middleware order is deliberate and load-bearing.** FastAPI executes middleware in *reverse* registration order, so the registration order in `main.py` is inverted on purpose (`TrustedHostMiddleware` added last → runs first; security headers added first → runs last). Read the comments there before reordering. `SlowAPIMiddleware` enforces the global rate limit; health endpoints are effectively exempted via a very high limit (1000/minute) rather than a true bypass.

`TrustedHostMiddleware` is outermost — ahead of even CORS — because starlette's router rebuilds redirect targets from the `Host` header and `redirect_slashes` defaults to true, so any trailing-slash path (including unauthenticated `/health/`) otherwise returns a 307 whose `Location` points at a caller-supplied host. `allowed_hosts` defaults to `["*"]` for local dev but `Settings` **rejects `"*"` whenever `app_env != "development"`** (staging and production both — broader than `enable_mock_tools`'s production-only guard, though the same fail-fast shape). Patterns use starlette's `*.example.com` form; a Django-style `.example.com` is rejected at config time because starlette would treat it as a literal hostname and silently match nothing. `www_redirect=False` is set so no rejection path can emit a Host-derived `Location` at all.

The global `Exception` handler returns a generic `ErrorResponse` (never leaks internals) and logs full detail in a **background task** so logging latency never blocks the response.

### Store & LLM factories

- `app/stores/factory.py` owns the settings → implementation mapping (`build_session_store`, `build_vector_store`) and the startup `dry_run_stores()` probe. To add a backend, extend the factory — don't hard-code a store in `lifespan`. Only genuinely external backends are probed (Redis, Ollama embeddings); in-memory and embedded Chroma are skipped.
- `app/llm/factory.py::build_fallback_model()` wraps `build_model()` in a `pydantic_ai` `FallbackModel` — **always**, even with zero configured fallbacks (a chain of one), so misconfiguration is found at startup. It imports `build_model` lazily inside the function because `chat_agent.py` imports `supports_native_output` from it (circular import otherwise).
- `supports_native_output()` reads capability from a `FallbackModel`'s *primary* model, since `FallbackModel.profile` raises `NotImplementedError`.

### Output-type gating (NativeOutput)

`build_chat_agent()` picks `output_type` from the model's profile: `NativeOutput(ChatOutput)` when `supports_json_schema_output` is true, plain `str` otherwise. Everything downstream must handle **both** — see the `isinstance(output, ChatOutput)` branches in `app/api/v1/agent.py` and `evals/runner.py`. In the SSE stream this matters more: under `NativeOutput` the streamed text is the *raw JSON envelope*, so text deltas are suppressed and the parsed `ChatOutput.reply` is emitted as one `Token` at the `End` node instead. The `test_model` fixture pins `supports_json_schema_output=False` to stay on the plain-text path.

`build_model()` in `app/agents/chat_agent.py` remains the single-model builder and source of truth for LiteLLM routing: the internal `"provider:model"` format (e.g. `openai:gpt-4o`) becomes LiteLLM's `"provider/model"`. **Ollama base-URL gotcha:** LiteLLM auto-appends `/v1`, so the chat model uses `http://localhost:11434` *without* `/v1`, while `OllamaEmbeddingVectorStore` calls Ollama directly and needs the full `/v1` URL. Don't "fix" this asymmetry — it's correct.

### Agent guardrails

`app/agents/guardrails.py` wraps every agent run. `run_guarded()` (non-streaming) and `build_guarded_toolset()` (streaming) share one `_GuardedToolset` that enforces, per tool call: allow-list → approval hook → pre-side-effect token budget, recording every refusal into an `AuditTrail` that ships to the client in `ChatResponse.audit`. Outcomes use a closed `StopReason` vocabulary (`completed`/`max_iterations`/`budget_exceeded`/`denied`/`disallowed_tool`); native `UsageLimitExceeded` is mapped into it by `classify_usage_limit_exceeded()`.

Three non-obvious invariants:

- The install idiom is `agent.override(tools=[], toolsets=[guarded])`. `agent.toolsets` **re-includes** tools registered via `@agent.tool` regardless of `override(toolsets=...)`, so omitting `tools=[]` double-registers every direct tool.
- Refused tools stay **visible** in the model's schema rather than being hidden, so a hallucinated call still reaches `call_tool()` and gets audited.
- A stopped run persists **no** session history (the audit trail is the record of that turn) — only `stop_reason == "completed"` saves.

### Session ownership

Session ids are server-issued and self-authenticating: `app/services/session_service.py` mints `{principal.id}.{token}.{signature}` where the signature is an HMAC over the first two parts keyed by `session_signing_key`. `authorize_session()` verifies it with `secrets.compare_digest` and 403s on any malformed/foreign id. `principal.id` is `sha256(api_key)[:16]` (`app/security/principal.py`). **This design deliberately requires no store schema or lookup for ownership** — don't add an ownership table.

`POST /v1/agent/chat` mints a new id when none is presented; `POST /v1/agent/stream` can only *authorize* an existing one, because the SSE contract has no field to return a new id on. RAG queries are session-less.

### SSE contract

`app/patterns/sse.py` owns the wire format: a discriminated union of exactly 5 typed events (`step_started`, `tool_called`, `token`, `completed`, `error`), all `extra="forbid"` so payloads can't grow raw or sensitive fields. `parse_sse_events()` splits only on real SSE terminators (`\r\n|\r|\n`), never `str.splitlines()`, because pydantic leaves U+2028/U+2029 unescaped in JSON and `splitlines()` would treat them as frame boundaries.

`app/api/v1/_stream.py` owns lifecycle hardening and nothing else (no wire codec, no policy): the `sse_max_events` cap, `is_disconnected()` polling, terminal `error` event, `CancelledError` re-raise after cleanup, and a per-event `sse_send_timeout`. Two subtleties worth reading before touching it:

- `_drive_to_queue` runs the event generator to completion in **one** persistent task, communicating via `asyncio.Queue`. `Agent.iter()` holds an anyio cancel scope open across yields, and anyio requires the same task to enter and exit it — driving `__anext__()` through a fresh `asyncio.wait_for()` per iteration raises "Attempted to exit cancel scope in a different task".
- The heartbeat in `agent.py::_with_heartbeat` uses `asyncio.wait()`, never `asyncio.wait_for()`, so a heartbeat tick checks readiness without cancelling the in-flight event.

### Corrective RAG workflow

`app/workflows/corrective_rag.py` is a LlamaIndex `Workflow` with three `@step`s wired by events (`events.py`): **search → evaluate → synthesize**, looping back via a fresh `SearchEvent` when context is insufficient and retries remain. Per-run state lives in the event chain (`WorkflowState`), not on the instance, so instances are safely reusable. The class is composed from mixins split per the file-size policy: `rag_cache.py` (`ResultCacheMixin`), `rag_llm.py` (`LLMCallMixin`), `rag_prompts.py` (`PromptBuildingMixin`).

`run()` is overridden in `ResultCacheMixin` for a **TTL + LRU result cache** with thundering-herd protection: concurrent identical queries share one in-flight `asyncio.Future` instead of all executing. Keyed by `sha256(query|max_retries)`; cached dicts are always copied out to prevent caller mutation. Two timeout layers: `llm_agent_timeout` per LLM call (returns a safe fallback, no retry on timeout) and `rag_workflow_timeout` for the whole run (→ HTTP 504 in `app/api/v1/rag.py`).

Answers must be **grounded**: `citation.py::order_hits()` sorts deterministically by `(-score, chunk_id)` so citations are stable across runs, and `validate_citations()` rejects empty or dangling citation ids. Both failures become HTTP 502 in the route — never a silently ungrounded answer.

`rag_prompts.py` carries a long module-level security note: `html.escape()` around `<query>`/`<context>` is defense-in-depth only. LLMs decode entities, so escaping is *not* the prompt-injection boundary — role separation is. Read that note before treating escaping as a mitigation.

Because instances are reusable, `get_rag_workflow` (`app/deps/workflow.py`) caches one workflow per vector-store object in a `WeakKeyDictionary` (avoids leaks and `id()` reuse), guarded by a `threading.Lock`. `_get_cached_model` is a separate `lru_cache` keyed on model name + base URL (never the API key).

### Pluggable interfaces (Protocols)

Extend by implementing a `typing.Protocol`, not by subclassing (Constitution Principle 2):

- `VectorStore` (`app/stores/vector_store/protocol.py`) — `InMemoryVectorStore` (TF-IDF), `ChromaVectorStore`, `OllamaEmbeddingVectorStore`. Async `close()` is called in lifespan.
- `SessionStore` (`app/stores/session_store/protocol.py`) — `InMemorySessionStore` (LRU + TTL) and `RedisSessionStore`. Stores `pydantic_ai` `ModelMessage` objects.
- `Judge[T]` (`evals/graders.py`) — lets a golden-set run inject an independent judge model.

### Config & security (`app/config/`)

`Settings` is composed from domain mixins (`llm.py`, `store.py`, `security.py`, `observability.py`) and re-exported from the package, so `from app.config import Settings, get_settings` is stable. `extra="forbid"` means typos in `.env` fail fast. `get_settings()` is `@cache`d **and called at module level in `main.py`**, so tests must clear the cache (autouse fixtures in `tests/conftest.py` do). Secrets use `SecretStr`. Validators enforce: API-key and signing-key strength (16+ chars, no placeholders), cloud providers (`openai`/`anthropic`/`groq`) require `llm_api_key`, HTTPS for non-localhost URLs, `redis_session_store_enabled` requires `redis_url`, `vector_store_backend="ollama"` requires `embedding_model`, and `enable_mock_tools` is rejected when `app_env == "production"`.

**Mock tools are double-guarded:** registered only when `app_env != "production"` *and* `enable_mock_tools` is set, and the import itself is deferred (`app/agents/tools_mock.py`) so mock code can't run in production even if misconfigured.

**Never write a model id as a literal** (`"openai:gpt-4o"` in an assignment) anywhere in `app/` or `evals/` — always `Settings.llm_model`. Both a pre-commit pygrep hook and `tests/unit/test_no_hardcoded_model_ids.py` enforce this.

### Auth & rate limiting

`verify_api_key` (`app/deps/auth.py`) validates `X-API-Key` with `secrets.compare_digest` and **returns a `Principal`** — it's an identity dependency, not just a gate. Applied per-route via `Depends`, which keeps `/health*` unauthenticated; all `/v1/*` routes require it.

LLM-invoking routes (`/agent/chat`, `/agent/stream`, `/rag/query`) additionally carry `Depends(enforce_llm_rate_limit)`, a stricter configurable `llm_rate_limit` that deliberately reuses `app.state.limiter` so it shares the same Redis-or-memory storage as the global limit.

### Observability

`configure_logfire()` (`app/observability.py`) scrubs `prompt`/`tool_input`/`tool_output` by default; `log_sensitive_payloads=True` disables scrubbing and emits an `AUDIT:` warning. Any init failure is caught and logged — observability never blocks startup. Pydantic AI is instrumented even without a token (local dev traces).

### Health

`/health` is liveness (`{"status": "ok"}`). `/health/ready` runs concurrent live probes of the session store (Redis only), vector store (Ollama only), and LLM provider (a `max_tokens=1` request), returning 200 `ready` or 503 `not_ready` with a per-dependency `checks` map. Backends with no external dependency report `"skipped"`, not `"healthy"`.

### Evals harness (`evals/`)

`evals/graders.py` grades on two axes — Outcome (correctness/completeness) and Behavior (tool-use discipline/faithfulness) — each 1–5 or `"Unknown"` with a required rationale. The judge is injected separately from the agent under test to avoid self-evaluation bias. `evals/runner.py` loads `evals/golden/*.json` and exits non-zero when an axis aggregate falls below 3.0. It makes real LLM calls on both sides — pre-push only, never CI.

## Testing

Layers mirror the `mise` tasks: `tests/unit/` (hermetic), `tests/integration/` (real stores + `FunctionModel`), `tests/e2e/` (HTTP), `tests/benchmarks/`, `tests/local/` (Ollama-gated by the `ollama` marker).

- **Unit tests are hermetic.** An autouse fixture wraps anything under `tests/unit/` in `tests/support/hermetic.py::block_network()`, which intercepts `AF_INET`/`AF_INET6` `socket.connect` and raises `NetworkBlockedError`. A missed mock fails loudly and instantly. Other tiers are intentionally *not* blocked.
- **LLM substitute** is the `test_model` fixture (`FunctionModel` with `supports_json_schema_output=False`) — prefer it over `TestModel` in integration/e2e.
- **`build_test_settings(**overrides)`** in `conftest.py` builds an isolated `Settings` by passing explicit field values, bypassing env/`.env` lookups.
- **Autouse isolation fixtures**: `clear_settings_cache` (`get_settings.cache_clear()`), `clear_workflow_cache` (`_workflow_cache` + `_get_cached_model`), `test_env` (minimal valid env vars). Tests verifying cloud-provider key validation must `monkeypatch.delenv("LLM_API_KEY")` explicitly, since `test_env` sets it.
- **`EXPECT_LIVE_TESTS=N`** fails the session when the count of tests that actually reached phase `call` ≠ N — an anti-false-green guard for gated lanes that could otherwise silently skip everything.

### Repo guards (tests that assert on the repo, not the app)

Several unit tests enforce project rules and will fail on structural drift. Understand what they assert before working around them:

- `test_file_size_policy.py` — hard-fails any `app/**`/`tests/**` file at ≥1000 lines; warns on 500–999.
- `test_contract_drift.py` — README's normative fences must not show classes/fields/discriminators that no longer exist. Deliberately one-directional: the README may *omit* newer members, it may not *show* dead ones.
- `test_ci_workflows.py` — parses the workflow YAML; every `uses:` must be pinned to a full 40-char SHA.
- `test_no_hardcoded_model_ids.py`, `test_naming_conventions.py`, `test_pre_push_hook.py`, `test_expect_live_tests_plugin.py`, `test_block_network.py`.

## Conventions

- **File-size policy** (`.sdd/steering/file-size-policy.md`): <500 lines OK, 500–999 review for splitting, ≥1000 prohibited. Reference splits: `app/config/` (mixin per domain), `app/workflows/` (mixin per concern), `app/stores/*/` (protocol + one file per backend).
- **Docstrings**: Google style, required on all public symbols. Requirement IDs (`Req 4.6`) in docstrings trace back to `.sdd/specs/<feature>/spec.md`.
- **SDD workflow**: specs, reviews, steering, and constitution live under `.sdd/`. Active feature specs in `.sdd/specs/<feature>/`; the 7 core principles are in `.sdd/memory/constitution.md`.
- **Design docs** worth reading before related work: `docs/tool-design-conventions.md` (before adding a real agent tool), `docs/owasp-agentic-llm-mapping.md`, `docs/production_deployment.md`.
