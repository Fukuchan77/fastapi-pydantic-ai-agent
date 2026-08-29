# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Commands

All tooling runs through `mise` (which wraps `uv`). Check `mise.toml` before running bare tools.

```bash
mise run dev                 # uvicorn dev server, hot reload, :8000
mise run test                # full suite with coverage
mise run test:unit           # fast, no LLM/I/O
mise run test:integration    # real stores + FunctionModel LLM
mise run test:e2e            # full HTTP stack (AsyncClient)
mise run test:ci             # what PR CI runs: unit+integration+e2e + coverage gate
mise run test:benchmark      # latency/throughput/cache-hit benchmarks (-s)
mise run test:local          # requires a running Ollama instance (-m ollama)
mise run test:redis          # requires a reachable Redis server (-m redis)
mise run evals               # offline LLM-judge golden set; makes REAL LLM calls (pre-push only)
mise run lint                # ruff check + ty check (type checker is `ty`, NOT mypy)
mise run format              # ruff format
mise run audit               # pip-audit dependency vulnerability scan
mise run hooks:install       # install pre-commit hook
mise run build               # docker build
```

Run a single test: `uv run pytest tests/unit/stores/test_session_store.py::test_name -v`

## Code Style

- **Type checker**: `ty` (NOT mypy) — strict mode, no implicit `Any`
- **Linter**: Ruff with `S` (bandit), `ANN` (annotations), `D` (google-style docstrings), `B`, `SIM` — line length 100, Python 3.13+
- **Imports**: one per line (`force-single-line = true`), two blank lines after imports block
- **Docstrings**: Google style required on all public symbols; tests relax `S101`/`ANN`
- **No hardcoded model IDs**: Never assign `"openai:gpt-4o"` etc. as a literal — always use `Settings.llm_model`. Pre-commit hook and unit test enforce this.
- **File size**: <500 lines OK, 500–999 review splitting, ≥1000 **prohibited** — see `.sdd/steering/file-size-policy.md`
- **No direct `os.environ` reads**: All env access goes through `Settings` / `get_settings()`. Constitution Principle 4.
- **`evals/`** is production-linted code (not a scratch directory) — Ruff + `ty` cover it.
- **`filterwarnings = ["error::DeprecationWarning"]`** in `pyproject.toml` — any deprecation warning from any module is a hard test error. Re-census on every pydantic-ai constraint bump, **and on every Python version change**.
- **Python is pinned to 3.13** (`.python-version` + `mise.toml`), not merely floored by `requires-python`. 3.14 deprecates `asyncio.iscoroutinefunction`, which `starlette` 0.52.x and `slowapi` 0.1.10 still call and the `starlette<1.0` pin prevents upgrading away from; with warnings-as-errors that is 63 test failures. Guarded by `tests/unit/test_python_version_pin.py`.

## Architecture

`app/main.py` is the composition root — `create_app()` (factory) is used everywhere; `app = create_app()` at module bottom is the uvicorn entrypoint. Tests always call `create_app(settings=..., model=test_model)` directly, never import the module-level `app` singleton.

**`app/lifespan.py::build_lifespan` owns all shared singletons** (`settings`, `http_client`, `vector_store`, `session_store`, `llm_model` — the resolved model shared by the chat agent and the RAG workflow — `chat_agent`, `cleanup_task`) on `app.state`; `create_app` keeps middleware, routers, and the global `Exception` handler. Startup is fail-fast: stores are built, probed via `dry_run_stores()`, and the `FallbackModel` chain is built eagerly — a misconfigured provider or unreachable Redis/Ollama kills startup. A test-injected `model` bypasses the fallback build entirely. A failure partway through `_startup` runs the same `_shutdown()` the normal path uses (each close isolated by `_close_quietly`) before re-raising unchanged.

**One flat error envelope**: `app/api/errors.py` registers on `starlette.exceptions.HTTPException`, **not** `fastapi.HTTPException` — starlette resolves handlers up the MRO, so registering on the subclass would miss the 404/405 the router raises. Validation failures drop the per-field `detail` so responses never echo request content. 413/429 are emitted by their own middleware/handler and never reach `HTTPException`.

**CORS is this project's own middleware** (`app/middleware/cors.py`), not starlette's: it merges `Origin` into an existing `Vary` header instead of replacing it, and a disallowed origin gets no CORS headers rather than an error.

**Middleware order is inverted** — FastAPI executes in *reverse* registration order. `TrustedHostMiddleware` is added last so it runs first (ahead of CORS); SecurityHeaders added first so it runs last. `RequestSizeLimitMiddleware` is registered *before* `RequestIDMiddleware` so the latter runs first and stamps `X-Request-ID` even on 413 responses. Do not reorder without reading the comments in `main.py`.

**Host validation** — `allowed_hosts` defaults to `["*"]` for dev, but `Settings` rejects `"*"` whenever `app_env != "development"` (staging and production both): starlette rebuilds redirect `Location` from the `Host` header and `redirect_slashes` is on, so a wildcard allow-list is an unauthenticated open-redirect. Wildcards use starlette's `*.example.com` form, not Django's `.example.com`.

**Environment vocabulary**: `app_env` is a closed `Literal["development", "staging", "production"]` narrowed from `str`, no normalization — it gates the host-wildcard rejection above, `enable_mock_tools`, and the `trust_proxy_headers` startup warning.

**Transport security & CSP**: `SecurityHeadersMiddleware` (`app/middleware/security_headers.py`) computes HSTS (`hsts_max_age`/`hsts_include_subdomains`) and CSP per request, not at construction. HSTS is omitted entirely on non-HTTPS responses — asserting "always HTTPS" on a plaintext response would be a lie. CSP is relaxed (CDN/font hosts, `'unsafe-inline'`) only for the live docs paths (`docs_url`/`redoc_url`/`swagger_ui_oauth2_redirect_url`), never a hardcoded path list.

**LLM model format**: `"provider:model"` (e.g. `openai:gpt-4o`) — converted to LiteLLM's `"provider/model"` internally by `build_model()` in `app/agents/chat_agent.py`.

**`FallbackModel` always wraps** even with zero configured fallbacks (a chain of one), so misconfiguration is caught at startup. `supports_native_output()` reads capability from the *primary* model only because `FallbackModel.profile` raises `NotImplementedError`.

**Ollama URL gotcha**: LiteLLM auto-appends `/v1`, so the chat model uses `http://localhost:11434` (no `/v1`). `OllamaEmbeddingVectorStore` calls Ollama directly and needs `http://localhost:11434/v1`. Do not "fix" this asymmetry.

**NativeOutput gating**: `build_chat_agent()` picks `NativeOutput(ChatOutput)` when `supports_json_schema_output` is true, plain `str` otherwise. Under `NativeOutput`, SSE text deltas are suppressed and the parsed `ChatOutput.reply` is emitted as one `Token` at the `End` node. Every downstream consumer (route handler, evals runner) must handle both paths.

**`end_strategy="early"` is pinned explicitly** in `build_chat_agent()` — v2 flipped `Agent.__init__`'s default to `"graceful"`, which would keep executing pending tool calls after the final result and shift guardrail tool-call counts, audit-trail contents, and token-budget accounting. The RAG agents omit it deliberately (no function tools registered). Its test asserts on the mocked constructor kwargs, since a post-construction read can't distinguish a pin from a matching default.

**Agent guardrails** (`app/agents/guardrails.py`) wrap every run in one `_GuardedToolset`: allow-list → approval hook → token budget, every refusal recorded into an `AuditTrail`. Outcomes use a closed `StopReason` vocabulary (`completed`/`max_iterations`/`budget_exceeded`/`denied`/`disallowed_tool`); native `UsageLimitExceeded` is mapped in by `classify_usage_limit_exceeded()`. Install idiom is `agent.override(tools=[], toolsets=[guarded])` — omitting `tools=[]` double-registers direct tools because `agent.toolsets` re-includes `@agent.tool` registrations regardless of `override(toolsets=...)`. Refused tools remain **visible** in the model's schema (not hidden), so hallucinated calls still reach `call_tool()` and get audited. A stopped run persists **no** session history — only `stop_reason == "completed"` saves. `classify_usage_limit_exceeded()` keys off the exception's message string (no structured attribute exists); v2 widened the raise sites 7 → 10 (`check_cost`, `check_per_request_input_tokens`, plus a `cost_limit` branch in `check_before_request`), so `test_usage_limit_templates.py` matches by prefix **plus** keying-substring containment, not exact template text. Toolset composition is pinned at **both** install sites (`run_guarded` and the SSE stream) — each builds its own toolset, so one test proves nothing about the other.

**v2 output-tool events**: v2 classifies output-tool invocations as sibling event kinds (`OutputToolCallEvent`/`OutputToolResultEvent`, not `FunctionToolCallEvent` subclasses), so they would stop surfacing as `tool_called` if the output mode ever became tool-based — inert today under `NativeOutput`. Function tools still raise `FunctionToolCallEvent`; the wire contract still emits exactly the five typed event kinds.

**v2 usage-instrumentation rename**: `logfire.instrument_pydantic_ai()` is called with no arguments, so agent-**run** spans now emit cumulative token attributes as `gen_ai.aggregated_usage.*` instead of `gen_ai.usage.*` (nested `chat` spans unchanged), and the data format moved 2 → 5. No `app/` code changed for this — see `docs/pydantic-ai-v2-behaviour-notes.md` before building a Logfire dashboard or query.

**SSE `_drive_to_queue` gotcha**: `Agent.iter()` holds an anyio cancel scope open across yields; anyio requires the same task to enter and exit it — driving `__anext__()` through a fresh `asyncio.wait_for()` per iteration raises "Attempted to exit cancel scope in a different task". The driver runs as one persistent task communicating via `asyncio.Queue`.

**Settings cache**: `get_settings()` is `@cache`d and called at module level in `main.py`. Tests must call `get_settings.cache_clear()` after patching env vars — the autouse `clear_settings_cache` fixture in `tests/conftest.py` does this automatically. `Settings` has `extra="forbid"` — typos in `.env` cause a `ValidationError` at startup.

**Mock tools are double-guarded**: registered only when `app_env != "production"` AND `enable_mock_tools` is set; import of `app/agents/tools_mock.py` is deferred. Never enable in production config.

**RAG sufficiency is structured, not parsed**: `_eval_agent`'s output type is `RelevanceVerdict` (`sufficient: bool` + non-empty `rationale`) — never text-match the reply. Nested retry budgets: pydantic-ai `retries={"output": ...}` for validation failures within one call, `_run_agent_with_retry` for transient failures across calls; a timeout returns the safe `sufficient=False` fallback with no retry.

**`/health/ready` probes are cached** by `ReadinessProbeCache` (`app/api/health.py`, per-app, TTL `readiness_probe_cache_ttl`, default 10s). A cost control, not a latency one: the route is unauthenticated and its LLM probe is a billable provider request, so uncached it turns inbound volume into outbound provider volume at the global 1000/minute limit — outside `llm_rate_limit`. Any new unauthenticated route touching a metered dependency needs its own bound.

**RAG workflow cache**: keyed by `sha256(query|max_retries|vector_store.generation)` — the generation counter invalidates pre-ingest entries without disturbing in-flight requests; thundering-herd protection via `_pending_futures`; `get_rag_workflow` caches per vector-store using a `WeakKeyDictionary`. The autouse `clear_workflow_cache` fixture resets that cache before/after every test.

**RAG uses the shared model chain**: `get_rag_workflow` reads `app.state.settings` and `app.state.llm_model` — the same `FallbackModel` chain the chat agent uses — instead of rebuilding a model. Either singleton absent fails the request with a flat-envelope 503 (`code="DEPENDENCY_NOT_INITIALIZED"`), never a silent fallback to process-global settings.

**Session ownership without a lookup table**: session ids are `{principal.id}.{token}.{signature}` (HMAC-signed). `authorize_session()` verifies with `secrets.compare_digest` and 403s on malformed/foreign ids. `POST /v1/agent/stream` can only *authorize* an existing id — no new id can be returned on that endpoint. **Do not add an ownership table** — the design is stateless-by-intent.

**`verify_api_key`** (`app/deps/auth.py`) is an identity dependency, not just a gate — it validates `X-API-Key` and **returns a `Principal`**. `principal.id` is `sha256(api_key)[:16]`, which seeds session ids. It resolves settings through `Depends(get_request_settings)` (`app/deps/settings.py`), **not** `get_settings` — see the settings-source rule below.

**`AgentDeps.principal` is bound at the route, not in the dependency**: agent routes call `bind_principal(deps, principal.id)` because resolving it inside `get_agent_deps()` would import `app.deps.auth` and close a cycle (`app.deps` → `app.deps.workflow` → `app.workflows.corrective_rag` → `app.agents.chat_agent` → `app.agents.deps`). Every `AuditRecord` is attributed through that field, so a new agent route must bind it.

**One `Settings` per decision**: request-path code reads `app.state.settings` (via `app/deps/settings.py::get_request_settings` for `Depends` sites, or `_resolve_settings()` inside `app/middleware/rate_limit.py`), never process-global `get_settings()`. `create_app(settings=...)` injects an explicit instance, so mixing the two sources means one half of a decision comes from the injection and the other from the environment — `get_client_identifier` once read `trusted_proxies` that way while `enforce_llm_rate_limit` read the injected limit. `get_settings()` survives only as a fallback for an app whose lifespan never populated `app.state`.

**`Settings.api_key` holds exactly one key**, so every caller derives the same `principal.id`. Two subsystems are correct only because of that and must be revisited when a second key lands — `/v1/rag/ingest` writes into the one process-wide vector store with no principal binding, and the RAG result cache is keyed on `(query, max_retries, generation)` alone. The module docstring of `app/security/principal.py` is the canonical statement.

**Redis session key prefix cut over to `"session:v2:"`** (`RedisSessionStore.DEFAULT_KEY_PREFIX`, was `"session:"`) alongside the pydantic-ai v2 bump, so a v2 instance never deserializes v1-written history. The factory passes no prefix, so that class default *is* the production prefix. Pre-cutover sessions are dropped by design — no migration and no deletion path; old keys expire on their own write-time TTL. During a cutover the deserialization-warning rate is the only signal of a wire-format mismatch (`docs/production_deployment.md`).

**Session trimming**: `trim_history()` (`app/stores/session_store/_trim.py`) is a pure function shared by both `SessionStore` backends, bounded by `session_max_messages` (default `1000`). Cuts land only between messages, never orphan a retained tool-call pair, and always keep `messages[0]` (the system prompt) — which is why the result can be `max_messages + 1` long, not exactly `max_messages`.

**Pluggable stores**: implement `typing.Protocol` in `app/stores/*/protocol.py`; register in `app/stores/factory.py`; wire via `lifespan`. Never subclass a concrete backend.

## Testing

- **Unit tests** (`tests/unit/`) have real network blocked via `tests/support/hermetic.py` — any missed mock raises `NetworkBlockedError` immediately
- **LLM substitute**: use `FunctionModel` (with `supports_json_schema_output=False` profile) from `tests/conftest.py::test_model` fixture — never use `TestModel` in integration/e2e layers
- **`build_test_settings(**overrides)`** helper in `conftest.py` constructs isolated `Settings` without relying on env vars
- **`asyncio_mode = "auto"`**: all coroutine tests are collected and awaited automatically — no `@pytest.mark.asyncio` needed
- **`EXPECT_LIVE_TESTS=N`** env var: session fails if the actual executed test count ≠ N — used to guard gated test lanes from silent skips
- **`chroma` marker**: opt-in (`RUN_CHROMA_INTEGRATION_TESTS` env var); downloads a Hugging Face embedding model — do not run routinely: `RUN_CHROMA_INTEGRATION_TESTS=1 EXPECT_LIVE_TESTS=6 uv run pytest tests/integration/test_chroma_query_with_scores.py` (`6` = `tests/support/chroma.py::CHROMA_LIVE_TEST_COUNT`)
- **`redis` marker**: gated on live reachability, not an opt-in var — `tests/support/redis.py::redis_reachable()` probes a real server and the lane skips (never fails) when none answers: `EXPECT_LIVE_TESTS=7 mise run test:redis` (`7` = `tests/support/redis.py::REDIS_LIVE_TEST_COUNT`)
- **`ollama` marker**: applied per test *function* across every module in `tests/local/`, so `tests/support/ollama.py::OLLAMA_LIVE_TEST_COUNT` is a sum, not one module's count. `test_local_test_gating.py` pins it against the pre-push hook literal and the prose restatements in `CLAUDE.md`/`AGENTS.md` — raise all three in lockstep; `docs/adapter-probe-report*.md` deliberately keeps its own run's count
- Tests for cloud-provider API key validation must `monkeypatch.delenv("LLM_API_KEY")` explicitly (the autouse fixture sets it by default)

### Repo-guard tests (assert on project structure, not app behavior)

These fail on structural drift — understand what they assert before bypassing:

- `test_file_size_policy.py` — hard-fails any `app/**`/`tests/**` file ≥1000 lines; warns on 500–999
- `test_contract_drift.py` — README's normative fences must not show classes/fields that no longer exist. **One-directional**: README may *omit* newer members, but may not *show* dead ones
- `test_ci_workflows.py` — every `uses:` in GitHub Actions YAML must be pinned to a 40-char SHA
- `test_pydantic_ai_api_lock.py` — subset-only lock on the pydantic-ai symbols/params/fields/kinds this project uses (no `app/` imports), so a dependency upgrade fails here by name. Add newly relied-upon symbols to its tables. `TestAntiFalseGreen` guards against tables being silently emptied.
- `test_config_dependency_bounds.py` — every production dependency must declare an upper bound
- `test_no_hardcoded_model_ids.py`, `test_naming_conventions.py`, `test_pre_push_hook.py`, `test_expect_live_tests_plugin.py`, `test_block_network.py`, `test_pytest_config.py`, plus the gating guards: `test_chroma_test_gating.py`, `test_local_test_gating.py`, `test_docker_deployment_gating.py`, `test_redis_test_gating.py`

## Dependency Pins That Are Load-Bearing

`fastapi<0.137` and `starlette<1.0` in `pyproject.toml` both exist because the newer version **silently disables the global rate limit** (router flattening / slowapi incompatibility) — read the comments there first; `tests/e2e/test_rate_limiting_enforcement.py` is the canary. The starlette pin is also why `mise run audit` carries `--ignore-vuln` entries: each is app-layer-mitigated or unreachable here. Re-run `uv run pip-audit` without them after any starlette bump.

A separate `--ignore-vuln` group covers 3 CVEs on `chromadb` 0.6.3, held below 1.0 as a shelved major. None has a fix release (`fixed_in: []`, and the affected range covers even the latest 1.5.9), so upgrading chromadb would not close them — all three need ChromaDB's server (multi-tenant HTTP API/RBAC/`trust_remote_code`), and this codebase only ever uses the embedded client (`app/stores/vector_store/chroma.py`), so they're unreachable today. They become reachable the moment anything switches to `chromadb.HttpClient` — see the note beside that example in `docs/production_deployment.md`.

`pydantic-ai-litellm` is pinned `>=0.2.3,<0.3.0` in `pyproject.toml` — capped below its next **minor**, not its next major: it's a 0.x package depending on six private pydantic-ai APIs, so its minors are its breaking releases and `<1.0` would admit 0.3.x–0.9.x unreviewed. Mirrors how `fastapi` is handled for 0.x versioning; `tests/unit/test_pydantic_ai_api_lock.py` is what catches such a breakage, not the pin.

A private-API coupling, not a version bound: the rate-limit-exceeded handler (`app/middleware/rate_limit.py`) delegates 429 header construction (`X-RateLimit-*`, delay-seconds `Retry-After`) to slowapi's `Limiter._inject_headers` — a leading-underscore method with no compatibility guarantee. A slowapi upgrade needs re-verification against `tests/unit/test_middleware_rate_limit_global_envelope.py` AND `tests/unit/middleware/test_rate_limit_retry_after.py`.

**`rate_limit_exceeded_handler` must be `def`, not `async def`** — slowapi's `SlowAPIMiddleware` uses `inspect.iscoroutinefunction` and silently swaps an `async` handler for its own default `{"error": ...}` handler, bypassing the flat error envelope. The handler body has no `await`, so sync is correct. Do not convert it to `async def`.

## CI Gating

- **PR CI** (`.github/workflows/pr.yml`): lint → `test:ci` → `test:redis` → `audit`. Never runs live LLM or Ollama tests, nor the `chroma`-marked tests (they self-skip unless `RUN_CHROMA_INTEGRATION_TESTS` is set, since they download a Hugging Face embedding model). The `redis`-marked lane does run here — CI starts a `redis:7-alpine` service container and sets `EXPECT_LIVE_TESTS=7` so a broken container fails the run instead of passing as a silent zero-collected green. `asyncio_mode = "auto"` means an unmarked coroutine test in this lane still runs instead of silently passing unawaited.
- **Nightly** (`.github/workflows/security.yml`): `pip-audit` + gitleaks, cron `37 3 * * *`. Steps run sequentially — a red `pip-audit` means gitleaks never runs that night, which hid 365 pre-existing findings for months (all `main`'s history) until `pip-audit` was fixed on 2026-08-29. All 365 are confirmed false positives (dummy `test-`/`sk-test-` keys in `tests/**` fixtures, 4 placeholder curl `-H "X-API-Key: ..."` lines in `README.md`); none in `app/`. `.gitleaksignore` lists them by exact `<commit>:<file>:<rule>:<line>` fingerprint (from `gitleaks detect --source . --report-format json`) — fingerprint-scoped, not text- or path-scoped, so a new commit reusing the same dummy string still gets flagged (verified). Regenerate the same way to add new entries; never hand-edit existing lines.
- **pre-commit** (`.pre-commit-config.yaml`): gitleaks, `pip-audit`, `no-hardcoded-model-id` pygrep, `real-tool-conventions-guard` (fires on any non-mock `@agent.tool` under `app/agents/`, forcing review of `docs/tool-design-conventions.md`).
- **pre-push** (`.githooks/pre-push`, opt-in via `git config core.hooksPath .githooks`): probes Ollama, runs `EXPECT_LIVE_TESTS=6 mise run test:local` + `evals` when reachable (the pinned count guards against a lane that silently collects zero live cases), warns and lets the push through when not.
- **Dependabot** (`.github/dependabot.yml`): weekly `uv` + `github-actions`, minors/patches grouped. Its `ignore:` list blocks the forbidden bumps — `starlette` majors and `fastapi` **minors and majors** (Dependabot reads fastapi's 0.x releases by patch position, so `0.136 → 0.137` is a minor) — plus `chromadb`/`redis` majors, shelved pending a client-compatibility pass. Guarded by `tests/unit/test_dependabot_config.py`; `docs/dependency-runbook.md` is the accept/shelve process.

## Feature Status (`004-pydantic-ai-v2-unblock` complete; `003-pydantic-ai-v2-migration` sealed, both tracked under `.sdd/specs/`)

- No known active bugs. The `InMemorySessionStore` LRU-victim defect previously listed here is fixed — victim selection now comes from `self._store.keys()` (`app/stores/session_store/in_memory.py`, which carries the comment explaining why).
- `003` shipped units 1–9; its task 9 recorded the adapter-compatibility gate **FAILED**, so its tasks 10–12 (Requirements 9–11) are closed as superseded by `004` rather than completed. `004` re-executed that gate under its own Requirement 4 (run 1), recording it **PASSED** (evidence: `docs/adapter-probe-report-2026-08-13-run1.md`, cross-referencing the original 2026-08-11 finding at `docs/adapter-probe-report.md`), unblocking the v2 code migration, its behavioural pinning, and the Redis key-prefix cutover.
- **All eight of `004`'s change units (tasks 1–8) have shipped**; the only unchecked subtasks are 6.6–6.8, conditional gate-*failure* branches that never fired. No spec is in flight — new work opens a new one. `CLAUDE.md` and `AGENTS.md` are still edited **as a pair in one change unit** (`004` Req 8.2).
- `pydantic-ai-slim` moved to the 2.x line (`>=2.27.0,<3.0` in `pyproject.toml`) in `004`'s task 7, unblocked by task 6's recorded gate PASS; `pydantic-ai-litellm` bumped alongside it to `>=0.2.3,<0.3.0`.

## Adding a New Real Agent Tool

Adding a non-mock `@agent.tool` under `app/agents/` triggers the `real-tool-conventions-guard` pre-commit hook. Read `docs/tool-design-conventions.md` before committing.
