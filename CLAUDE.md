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

Lint/type config lives in `pyproject.toml`: Ruff is strict (`S` bandit, `ANN` annotations, `D` google-style docstrings, `SIM`, `B`), imports are one-per-line (`force-single-line`), line length 100, Python 3.13+. `ty` runs in strict mode (no implicit `Any`). Tests relax `S101`/`ANN`. Coverage gate is `fail_under = 80`. `asyncio_mode = "auto"`, so an unmarked coroutine test cannot pass unawaited. Pytest markers: `docker`, `benchmark`, `ollama`, `chroma`.

`mise run lint`/`format` cover `app/`, `evals/`, and `tests/` — `evals/` is production-linted code, not a scratch directory. (`ruff` runs over all three; `ty check` covers `app/` and `evals/` only, so a type error confined to `tests/` will not fail `lint`.)

### Dependency pins that are load-bearing

Two upper bounds in `pyproject.toml` exist because the *newer* version silently disables the global rate limit — read the comments there before bumping either:

- `fastapi<0.137` — 0.137 wraps included routers in `_IncludedRouter` instead of flattening them into `app.routes`; slowapi's `_find_route_handler` scans `app.routes` non-recursively, finds no `.endpoint`, and `_should_exempt` then treats **every** request as exempt. `tests/e2e/test_rate_limiting_enforcement.py` is the canary.
- `starlette<1.0` — slowapi 0.1.10 is incompatible with starlette 1.x (exception-handler misdispatch, and `SlowAPIMiddleware` stops emitting `X-RateLimit-*`).

A third bound is *wrong in the other direction*: `pydantic-ai-litellm>=0.2.3,<1.0`. It is a 0.x package that reaches into private pydantic-ai APIs, so `<1.0` admits unreviewed 0.3.x–0.9.x releases; the bound should be `<0.3.0`, matching how `fastapi` is handled for 0.x versioning. `tests/unit/test_pydantic_ai_api_lock.py` is what would catch the breakage, not the pin.

That starlette pin is why `mise run audit` carries an `--ignore-vuln` list: each ignored advisory is either mitigated at the app layer (`PYSEC-2026-161`, closed by `TrustedHostMiddleware`) or unreachable here (no `request.form()`, `HTTPEndpoint`, or `StaticFiles`). Re-run `uv run pip-audit` *without* `--ignore-vuln` after any starlette bump before renewing that list. `tests/unit/test_config_dependency_bounds.py` separately requires every production dependency to declare an upper bound at all.

### Gating: what runs where

- **PR CI** (`.github/workflows/pr.yml`): lint → `test:ci` → `audit`. Never runs live LLM or Ollama tests, nor the `chroma`-marked tests (they self-skip unless `RUN_CHROMA_INTEGRATION_TESTS` is set, since they download a Hugging Face embedding model). `asyncio_mode = "auto"` means an unmarked coroutine test in this lane still runs instead of silently passing unawaited.
- **Nightly** (`.github/workflows/security.yml`): `pip-audit` + gitleaks, cron `37 3 * * *`.
- **pre-commit** (`.pre-commit-config.yaml`): gitleaks, `pip-audit`, a pygrep `no-hardcoded-model-id` guard, and an inert `real-tool-conventions-guard` that fires the moment a non-mock `@agent.tool` appears under `app/agents/` (forcing a read of `docs/tool-design-conventions.md`).
- **pre-push** (`.githooks/pre-push`, enable with `git config core.hooksPath .githooks`): availability-gated — probes `${OLLAMA_BASE_URL}/api/tags`, runs `EXPECT_LIVE_TESTS=5 mise run test:local` + `evals` when reachable (the pinned count guards against a lane that silently collects zero live cases), warns and lets the push through when not.
- **Dependabot** (`.github/dependabot.yml`): weekly `uv` and `github-actions` updates. It has no ecosystem-level ignore rules, so it *can* propose the `fastapi`/`starlette` bumps the pins above deliberately forbid — read the pin comments before accepting one.

## Known active defects

Verified present in the code as of this branch (`003-pydantic-ai-v2-migration`), each specified for repair in `.sdd/specs/003-pydantic-ai-v2-migration/spec.md`. They are listed because each one makes a *documented* guarantee false — don't rediscover them, and don't build on the broken behaviour:

- **The global 429 escapes the flat envelope.** `rate_limit_exceeded_handler` in `app/middleware/rate_limit.py` is `async def`, and `SlowAPIMiddleware` reaches it through `sync_check_limits`, which swaps any coroutine handler for slowapi's own `_rate_limit_exceeded_handler` (`inspect.iscoroutinefunction`). So a globally rate-limited request gets `{"error": "Rate limit exceeded: ..."}` with slowapi-injected `X-RateLimit-*`. Fix is one keyword: `async def` → `def`.
- **The `Retry-After` computation is unreachable.** `RateLimitExceeded.__init__` only sets `status_code`/`detail`, never `.headers`, so `exc.headers` is always `None` — the handler's `X-RateLimit-Reset` branch never runs and every 429 it *does* emit (the per-route `enforce_llm_rate_limit` path) carries a constant `Retry-After: 60` and no `X-RateLimit-*` at all. Header construction belongs to the limiter (`limiter._inject_headers`), not to this handler.
- **`_get_cached_model` (`app/deps/workflow.py`) ignores its own arguments.** The body is `get_settings(); build_model(settings)` regardless of the `llm_model`/`llm_base_url` it keys `@lru_cache` on. `get_rag_workflow` compounds it by calling `get_settings()` instead of reading `req.app.state.settings`, so **RAG ignores `create_app(settings=...)` test injection** and rebuilds its own model rather than sharing the startup `FallbackModel` chain. The planned fix deletes `_get_cached_model` and publishes the resolved model as a new `app.state.llm_model` singleton — note that attribute does **not** exist yet.
- **`InMemorySessionStore` LRU eviction picks its victim from the wrong dict.** `in_memory.py` selects `min(self._last_access.items(), ...)`, but `get_history()` stamps `_last_access` even for a session id that was never saved. A ghost entry can therefore win the vote, and the `lru_session_id in self._store` guard then skips the eviction entirely — so `max_sessions` is a soft limit until the ghost's TTL expires. Victim selection should come from `self._store.keys()`.

## Architecture

A FastAPI framework demonstrating three agent patterns. `README.md` describes the original design; the code has since grown guardrails, session ownership, provider fallback chains, Redis/Chroma/Ollama backends, and an evals harness — trust the code over the README where they differ (a guard test keeps the README's *shown* examples honest; see Repo guards below).

### Composition root

`app/main.py::create_app(settings=None, model=None)` is the factory. `app = create_app()` at the module bottom is only the `uvicorn app.main:app` entrypoint — **tests always call `create_app(settings=..., model=test_model)`** rather than importing the module-level singleton, which is what lets them avoid monkeypatching env vars.

Responsibility is split: `create_app` owns middleware registration, routers, and the global `Exception` handler; **`app/lifespan.py::build_lifespan(settings, model)`** owns startup/shutdown *ordering* only. It populates `app.state` with the shared singletons everything else reads: `settings`, `http_client` (built by `app/http_client.py::build_http_client` — an `httpx.AsyncClient` over a custom `RetryTransport` that retries only transient 5xx `{500,502,503,504}` plus network errors; `RetryTransport` is re-exported from `app.main` for existing callers), `vector_store`, `session_store`, `chat_agent`, and a background `cleanup_task`. Startup is **fail-fast**: stores are built from settings, probed via `dry_run_stores()`, and the `FallbackModel` chain is built eagerly — a misconfigured provider or unreachable Redis/Ollama kills startup instead of surfacing on the first request. A test-injected `model` bypasses the fallback build entirely.

Three details in `lifespan.py` are the point of the split:

- `_startup` assigns each resource to `app.state` **as soon as it is built**, and `lifespan` wraps the whole call in `try/except` that runs the same `_shutdown(app)` the normal path uses before re-raising unchanged — so a failure partway through startup still releases everything already opened.
- `_shutdown` closes each store/client through `_close_quietly()`, which logs (at `error`, since a leak should alert) instead of raising, so one failing close can never skip the ones after it. Every step is `hasattr`-guarded because a startup failure may leave any attribute unset.
- `cleanup_loop` re-raises `CancelledError` but swallows and logs everything else so a transient store error can't kill the loop (which would leak sessions), and its interval is floored at `CLEANUP_INTERVAL_MIN = 300` even when `session_ttl` is tiny in tests.

Two startup warnings are deliberate, not noise: a wildcard in `cors_origins`, and `trust_proxy_headers=False` outside development (without it, `SecurityHeadersMiddleware` never emits HSTS behind a TLS-terminating proxy — see `docs/production_deployment.md`).

**Middleware order is deliberate and load-bearing.** FastAPI executes middleware in *reverse* registration order, so the registration order in `main.py` is inverted on purpose (`TrustedHostMiddleware` added last → runs first; security headers added first → runs last). Read the comments there before reordering. `SlowAPIMiddleware` enforces the global rate limit; health endpoints are effectively exempted via a very high limit (1000/minute) rather than a true bypass. `RequestSizeLimitMiddleware` (10 MiB) is registered *before* `RequestIDMiddleware` specifically so the latter runs first and stamps `X-Request-ID` even on the 413 it emits.

CORS is **this project's own** `app/middleware/cors.py`, not starlette's: it validates the origin against the allow-list, rejects `allow_credentials=True` with `allow_origins=["*"]` at construction, and merges `Origin` into any existing `Vary` header rather than replacing it (a handler may already have set `Vary` for `Accept-Encoding`). A disallowed origin is processed normally but simply gets **no** CORS headers.

`TrustedHostMiddleware` is outermost — ahead of even CORS — because starlette's router rebuilds redirect targets from the `Host` header and `redirect_slashes` defaults to true, so any trailing-slash path (including unauthenticated `/health/`) otherwise returns a 307 whose `Location` points at a caller-supplied host. `allowed_hosts` defaults to `["*"]` for local dev but `Settings` **rejects `"*"` whenever `app_env != "development"`** (staging and production both — broader than `enable_mock_tools`'s production-only guard, though the same fail-fast shape). Patterns use starlette's `*.example.com` form; a Django-style `.example.com` is rejected at config time because starlette would treat it as a literal hostname and silently match nothing. `www_redirect=False` is set so no rejection path can emit a Host-derived `Location` at all.

### One flat error envelope

Every error response is the same flat `ErrorResponse` shape (`message` + `code`), so clients need one parser:

- The global `Exception` handler stays in `main.py` (it also owns background-task logging): a generic `ErrorResponse` that never leaks internals, with full detail logged in a **background task** so logging latency never blocks the response.
- `app/api/errors.py::register_error_handlers(app)` covers everything raised as an exception. It registers on **`starlette.exceptions.HTTPException`, not `fastapi.HTTPException`** — starlette resolves handlers by walking the raised class's MRO upward, so a registration on the FastAPI subclass would never match the base class the *router* raises for 404/405, and those would fall back to the legacy `{"detail": ...}` body. `_render_detail()` accepts a string, a mapping, or anything else; `_DEFAULT_CODE_BY_STATUS` supplies the `code` when the raise site carries none. Validation failures deliberately drop FastAPI's per-field `detail` list so a response never echoes request content back.
- 413 and 429 never reach `HTTPException` — they're emitted directly by `RequestSizeLimitMiddleware` and by the 429 handler in `app/middleware/rate_limit.py`. **The 429 is the one hole in the envelope today**, and the two 429 producers do not agree: the global (middleware-enforced) limit does *not* use our handler at all, so its body is slowapi's `{"error": ...}`, while the per-route `enforce_llm_rate_limit` 429 does use it and is flat. See Known active defects below before writing anything that parses a 429.

### Security headers: HSTS and CSP

`SecurityHeadersMiddleware` (`app/middleware/security_headers.py`) computes `Strict-Transport-Security` and `Content-Security-Policy` per request rather than fixing them at construction:

- HSTS is built from `hsts_max_age`/`hsts_include_subdomains` and omitted entirely on a non-HTTPS request — asserting "always HTTPS" on a plaintext response would be a lie the header can't take back. Scheme comes from the ASGI scope as resolved by the server layer, never a forwarded header (ADR-5); `trust_proxy_headers` is only the startup-warning confirmation that `--forwarded-allow-ips`/`FORWARDED_ALLOW_IPS` is configured to match, and grants no trust itself.
- CSP has two policies: a strict default and a relaxed one (`'unsafe-inline'` plus the docs CDN/font hosts) that applies only to the interactive documentation UI. The docs paths are read off the live `FastAPI` app's `docs_url`/`redoc_url`/`swagger_ui_oauth2_redirect_url`, not a hardcoded path list — the OAuth2 redirect sub-path needs the relaxed policy too, which is why an exact `path == "/docs"` check would be insufficient.

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

### Session history trimming

`session_max_messages` (`Settings`, default `1000`, `Field(ge=2)`) bounds every `SessionStore.save_history()` through the pure `trim_history()` (`app/stores/session_store/_trim.py`), shared by both backends. Invariants: cuts fall only **between** messages, never inside one; the retained tail never orphans a tool-call pair (every retained `BaseToolReturnPart`/`RetryPromptPart(tool_name is not None)` closer keeps its opening `BaseToolCallPart` too, found by searching forward from the ideal cut); `messages[0]` (the persisted system prompt) is always retained, which is why `len(result)` can be `max_messages + 1` rather than exactly `max_messages`; the result is never empty and never holds a message with `parts == []`.

### SSE contract

`app/patterns/sse.py` owns the wire format: a discriminated union of exactly 5 typed events (`step_started`, `tool_called`, `token`, `completed`, `error`), all `extra="forbid"` so payloads can't grow raw or sensitive fields. `parse_sse_events()` splits only on real SSE terminators (`\r\n|\r|\n`), never `str.splitlines()`, because pydantic leaves U+2028/U+2029 unescaped in JSON and `splitlines()` would treat them as frame boundaries.

`app/api/v1/_stream.py` owns lifecycle hardening and nothing else (no wire codec, no policy): the `sse_max_events` cap, `is_disconnected()` polling, terminal `error` event, `CancelledError` re-raise after cleanup, and a per-event `sse_send_timeout`. Two subtleties worth reading before touching it:

- `_drive_to_queue` runs the event generator to completion in **one** persistent task, communicating via `asyncio.Queue`. `Agent.iter()` holds an anyio cancel scope open across yields, and anyio requires the same task to enter and exit it — driving `__anext__()` through a fresh `asyncio.wait_for()` per iteration raises "Attempted to exit cancel scope in a different task".
- The heartbeat in `agent.py::_with_heartbeat` uses `asyncio.wait()`, never `asyncio.wait_for()`, so a heartbeat tick checks readiness without cancelling the in-flight event.

### Corrective RAG workflow

`app/workflows/corrective_rag.py` is a LlamaIndex `Workflow` with three `@step`s wired by events (`events.py`): **search → evaluate → synthesize**, looping back via a fresh `SearchEvent` when context is insufficient and retries remain. Per-run state lives in the event chain (`WorkflowState`), not on the instance, so instances are safely reusable. The class is composed from mixins split per the file-size policy: `rag_cache.py` (`ResultCacheMixin`), `rag_llm.py` (`LLMCallMixin`), `rag_prompts.py` (`PromptBuildingMixin`).

`run()` is overridden in `ResultCacheMixin` for a **TTL + LRU result cache** with thundering-herd protection: concurrent identical queries share one in-flight `asyncio.Future` instead of all executing. Keyed by `sha256(query|max_retries|vector_store.generation)` — the store's generation counter is folded in so an ingest invalidates every pre-ingest entry without disturbing a request already in flight under the pre-ingest key (Req 2.1–2.3). Cached dicts are always copied out to prevent caller mutation. Two timeout layers: `llm_agent_timeout` per LLM call (returns a safe fallback, no retry on timeout) and `rag_workflow_timeout` for the whole run (→ HTTP 504 in `app/api/v1/rag.py`).

**The sufficiency decision is structured, never parsed from prose.** `_eval_agent`'s output type is `RelevanceVerdict` (`app/models/rag.py`: `sufficient: bool` + a non-empty `rationale`), so `evaluate` reads a validated field — do not reintroduce text matching on the reply. Two *nested* retry budgets govern it and cover different failures: pydantic-ai's own `retries={"output": ...}` retries output-validation failures inside a single `agent.run()`, while `rag_llm.py::_run_agent_with_retry` retries transient failures (network, 5xx) across separate calls with exponential backoff + jitter. A timeout returns the fallback **immediately** with no retry (a consistently slow LLM isn't transient), and the eval fallback is a safe `sufficient=False` verdict.

Answers must be **grounded**: `citation.py::order_hits()` sorts deterministically by `(-score, chunk_id)` so citations are stable across runs, and `validate_citations()` rejects empty or dangling citation ids. Both failures become HTTP 502 in the route — never a silently ungrounded answer.

`rag_prompts.py` carries a long module-level security note: `html.escape()` around `<query>`/`<context>` is defense-in-depth only. LLMs decode entities, so escaping is *not* the prompt-injection boundary — role separation is. Read that note before treating escaping as a mitigation.

Because instances are reusable, `get_rag_workflow` (`app/deps/workflow.py`) caches one workflow per vector-store object in a `WeakKeyDictionary` (avoids leaks and `id()` reuse), guarded by a `threading.Lock`. `_get_cached_model` is a separate `lru_cache` keyed on model name + base URL (never the API key).

### Pluggable interfaces (Protocols)

Extend by implementing a `typing.Protocol`, not by subclassing (Constitution Principle 2):

- `VectorStore` (`app/stores/vector_store/protocol.py`) — `InMemoryVectorStore` (TF-IDF), `ChromaVectorStore`, `OllamaEmbeddingVectorStore`. Async `close()` is called in lifespan.
- `SessionStore` (`app/stores/session_store/protocol.py`) — `InMemorySessionStore` (LRU + TTL) and `RedisSessionStore`. Stores `pydantic_ai` `ModelMessage` objects.
- `Judge[T]` (`evals/graders.py`) — lets a golden-set run inject an independent judge model.

### Config & security (`app/config/`)

`Settings` is composed from domain mixins (`llm.py`, `store.py`, `security.py`, `observability.py`) and re-exported from the package, so `from app.config import Settings, get_settings` is stable. `extra="forbid"` means typos in `.env` fail fast. `get_settings()` is `@cache`d **and called at module level in `main.py`**, so tests must clear the cache (autouse fixtures in `tests/conftest.py` do). Secrets use `SecretStr`. Validators enforce: API-key and signing-key strength (16+ chars, no placeholders), cloud providers (`openai`/`anthropic`/`groq`) require `llm_api_key`, HTTPS for non-localhost URLs, `redis_session_store_enabled` requires `redis_url`, `vector_store_backend="ollama"` requires `embedding_model`, and `enable_mock_tools` is rejected when `app_env == "production"`. `app_env` itself is a closed `Literal["development", "staging", "production"]` narrowed from `str` — no normalization, so `"Production"` fails validation rather than being silently coerced; it gates the `allowed_hosts` wildcard rejection above, `enable_mock_tools`, and the `trust_proxy_headers` startup warning.

**Mock tools are double-guarded:** registered only when `app_env != "production"` *and* `enable_mock_tools` is set, and the import itself is deferred (`app/agents/tools_mock.py`) so mock code can't run in production even if misconfigured.

**Never write a model id as a literal** (`"openai:gpt-4o"` in an assignment) anywhere in `app/` or `evals/` — always `Settings.llm_model`. Both a pre-commit pygrep hook and `tests/unit/test_no_hardcoded_model_ids.py` enforce this.

### Auth & rate limiting

`verify_api_key` (`app/deps/auth.py`) validates `X-API-Key` with `secrets.compare_digest` and **returns a `Principal`** — it's an identity dependency, not just a gate. Applied per-route via `Depends`, which keeps `/health*` unauthenticated; all `/v1/*` routes require it.

LLM-invoking routes (`/agent/chat`, `/agent/stream`, `/rag/query`) additionally carry `Depends(enforce_llm_rate_limit)`, a stricter configurable `llm_rate_limit` that deliberately reuses `app.state.limiter` so it shares the same Redis-or-memory storage as the global limit.

The rate-limit key is `get_client_identifier()`, which trusts `X-Forwarded-For` **only** when the immediate `request.client.host` is in `trusted_proxies` (an empty list means the header is never trusted) — otherwise a client could spoof its way around the limit. The `Limiter` sets `in_memory_fallback_enabled=True`, so a configured-but-unreachable Redis degrades to in-memory counting with a warning instead of failing every request.

### Observability

`configure_logfire()` (`app/observability.py`) scrubs `prompt`/`tool_input`/`tool_output` by default; `log_sensitive_payloads=True` disables scrubbing and emits an `AUDIT:` warning. Any init failure is caught and logged — observability never blocks startup. Pydantic AI is instrumented even without a token (local dev traces).

Stdlib logging is separate and configured first: `configure_logging()` (`app/logging_config.py`, called at the top of `_startup` before anything else logs) installs a single stdout handler with a `JSONFormatter` (one JSON object per line: `timestamp`, `level`, `logger`, `message`, `request_id`, plus `exc_info` when present) and a `RequestIDFilter` that reads `app.middleware.request_id.request_id_var`. That filter is why no call site inserts a request id by hand — log with a plain `logging.getLogger(__name__)` and correlation comes for free. It is idempotent by an "already has handlers" check on the root logger, which also means **a test or harness that configures logging first silently wins**. Level is `DEBUG` in development, `INFO` in staging/production.

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
- **Opt-in lanes are env-gated, and their expected counts are asserted.** The `chroma` marker needs `RUN_CHROMA_INTEGRATION_TESTS` (it downloads a Hugging Face embedding model, so don't run it routinely); `tests/support/chroma.py::CHROMA_LIVE_TEST_COUNT` is the number to pass as `EXPECT_LIVE_TESTS`, and a guard test fails when that constant drifts from the gated module. `tests/local/` and the Docker tests work the same way (`tests/support/ollama.py`, `tests/support/docker.py`).

### Repo guards (tests that assert on the repo, not the app)

Several unit tests enforce project rules and will fail on structural drift. Understand what they assert before working around them:

- `test_file_size_policy.py` — hard-fails any `app/**`/`tests/**` file at ≥1000 lines; warns on 500–999.
- `test_contract_drift.py` — README's normative fences must not show classes/fields/discriminators that no longer exist. Deliberately one-directional: the README may *omit* newer members, it may not *show* dead ones.
- `test_ci_workflows.py` — parses the workflow YAML; every `uses:` must be pinned to a full 40-char SHA.
- `test_pydantic_ai_api_lock.py` — locks the pydantic-ai / `pydantic-ai-litellm` surface this project uses (symbols, parameter names, dataclass fields, and *kinds* like `property`/`staticmethod`/`cached_property`), with **no `app/` imports**, so a dependency upgrade fails here naming the symbol instead of surfacing as an opaque app-test failure. Every assertion is subset-only — upstream *additions* must stay green — and `TestAntiFalseGreen` guards against the tables being silently emptied. When you start relying on a new pydantic-ai symbol, add it here.
- `test_config_dependency_bounds.py` — every production dependency must declare an upper bound.
- `test_no_hardcoded_model_ids.py`, `test_naming_conventions.py`, `test_pre_push_hook.py`, `test_expect_live_tests_plugin.py`, `test_block_network.py`, `test_pytest_config.py`, and the gating guards (`test_chroma_test_gating.py`, `test_local_test_gating.py`, `test_docker_deployment_gating.py`).

## Conventions

- **File-size policy** (`.sdd/steering/file-size-policy.md`): <500 lines OK, 500–999 review for splitting, ≥1000 prohibited. Reference splits: `app/config/` (mixin per domain), `app/workflows/` (mixin per concern), `app/stores/*/` (protocol + one file per backend).
- **No `os.environ` reads outside `Settings`** (Constitution Principle 4). Every env var is declared on a `Settings` mixin and reached via `get_settings()` or `app.state.settings`; there are currently zero direct reads in `app/`, and settings validation is what turns misconfiguration into a startup error.
- **Docstrings**: Google style, required on all public symbols. Requirement IDs (`Req 4.6`) in docstrings trace back to `.sdd/specs/<feature>/spec.md`.
- **SDD workflow**: specs, reviews, steering, and constitution live under `.sdd/`. Active feature specs in `.sdd/specs/<feature>/`; the 8 core principles are in `.sdd/memory/constitution.md` (v1.1.0 — paths there match the current tree, and Governance requires a refactor to update any path it moves in the same change). Two smaller directories are retrospective, not planning: `.sdd/mistakes/` (per-feature post-mortems) and `.sdd/patterns/` (techniques worth reusing) — append to them rather than re-deriving a lesson already recorded.
- **The current feature (`003-pydantic-ai-v2-migration`) is specified but unimplemented** — `spec.json` is at phase `tasks-generated` with all three approvals, and all 90 tasks in `tasks.md` are open. Its tasks are boundary-scoped (`_Boundary:_` lists the only files a task may touch) and several of them own edits to *this file*, so check `tasks.md` before making an architectural change here: the change may already be assigned, sequenced, and required to ship behind a failing test.
- **Design docs** worth reading before related work: `docs/tool-design-conventions.md` (before adding a real agent tool), `docs/owasp-agentic-llm-mapping.md`, `docs/production_deployment.md`.
