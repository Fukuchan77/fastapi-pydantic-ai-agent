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

## Architecture

`app/main.py` is the composition root — `create_app()` (factory) is used everywhere; `app = create_app()` at module bottom is the uvicorn entrypoint. Tests always call `create_app(settings=..., model=test_model)` directly, never import the module-level `app` singleton.

**`app/lifespan.py::build_lifespan` owns all shared singletons** (`settings`, `http_client`, `vector_store`, `session_store`, `llm_model` — the resolved model shared by the chat agent and the RAG workflow — `chat_agent`, `cleanup_task`) on `app.state`; `create_app` keeps middleware, routers, and the global `Exception` handler. Startup is fail-fast: stores are built, probed via `dry_run_stores()`, and the `FallbackModel` chain is built eagerly — a misconfigured provider or unreachable Redis/Ollama kills startup. A test-injected `model` bypasses the fallback build entirely. A failure partway through `_startup` runs the same `_shutdown()` the normal path uses (each close isolated by `_close_quietly`) before re-raising unchanged.

**One flat error envelope**: `app/api/errors.py` registers on `starlette.exceptions.HTTPException`, **not** `fastapi.HTTPException` — starlette resolves handlers up the MRO, so registering on the subclass would miss the 404/405 the router raises. Validation failures drop the per-field `detail` so responses never echo request content. 413/429 are emitted by their own middleware/handler and never reach `HTTPException`.

**CORS is this project's own middleware** (`app/middleware/cors.py`), not starlette's: it merges `Origin` into an existing `Vary` header instead of replacing it, and a disallowed origin gets no CORS headers rather than an error.

**Middleware order is inverted** — FastAPI executes in *reverse* registration order. `TrustedHostMiddleware` is added last so it runs first (ahead of CORS); SecurityHeaders added first so it runs last. Do not reorder without reading the comments in `main.py`.

**Host validation** — `allowed_hosts` defaults to `["*"]` for dev, but `Settings` rejects `"*"` whenever `app_env != "development"` (staging and production both): starlette rebuilds redirect `Location` from the `Host` header and `redirect_slashes` is on, so a wildcard allow-list is an unauthenticated open-redirect. Wildcards use starlette's `*.example.com` form, not Django's `.example.com`.

**Environment vocabulary**: `app_env` is a closed `Literal["development", "staging", "production"]` narrowed from `str`, no normalization — it gates the host-wildcard rejection above, `enable_mock_tools`, and the `trust_proxy_headers` startup warning.

**Transport security & CSP**: `SecurityHeadersMiddleware` (`app/middleware/security_headers.py`) computes HSTS (`hsts_max_age`/`hsts_include_subdomains`) and CSP per request, not at construction. HSTS is omitted entirely on non-HTTPS responses — asserting "always HTTPS" on a plaintext response would be a lie. CSP is relaxed (CDN/font hosts, `'unsafe-inline'`) only for the live docs paths (`docs_url`/`redoc_url`/`swagger_ui_oauth2_redirect_url`), never a hardcoded path list.

**LLM model format**: `"provider:model"` (e.g. `openai:gpt-4o`) — converted to LiteLLM's `"provider/model"` internally by `build_model()` in `app/agents/chat_agent.py`.

**`FallbackModel` always wraps** even with zero configured fallbacks (a chain of one), so misconfiguration is caught at startup. `supports_native_output()` reads capability from the *primary* model only because `FallbackModel.profile` raises `NotImplementedError`.

**Ollama URL gotcha**: LiteLLM auto-appends `/v1`, so the chat model uses `http://localhost:11434` (no `/v1`). `OllamaEmbeddingVectorStore` calls Ollama directly and needs `http://localhost:11434/v1`. Do not "fix" this asymmetry.

**NativeOutput gating**: `build_chat_agent()` picks `NativeOutput(ChatOutput)` when `supports_json_schema_output` is true, plain `str` otherwise. Under `NativeOutput`, SSE text deltas are suppressed and the parsed `ChatOutput.reply` is emitted as one `Token` at the `End` node. Every downstream consumer (route handler, evals runner) must handle both paths.

**Agent guardrails** (`app/agents/guardrails.py`) wrap every run in one `_GuardedToolset`: allow-list → approval hook → token budget, every refusal recorded into an `AuditTrail`. Outcomes use a closed `StopReason` vocabulary (`completed`/`max_iterations`/`budget_exceeded`/`denied`/`disallowed_tool`); native `UsageLimitExceeded` is mapped in by `classify_usage_limit_exceeded()`. Install idiom is `agent.override(tools=[], toolsets=[guarded])` — omitting `tools=[]` double-registers direct tools because `agent.toolsets` re-includes `@agent.tool` registrations regardless of `override(toolsets=...)`.

**SSE `_drive_to_queue` gotcha**: `Agent.iter()` holds an anyio cancel scope open across yields; anyio requires the same task to enter and exit it — driving `__anext__()` through a fresh `asyncio.wait_for()` per iteration raises "Attempted to exit cancel scope in a different task". The driver runs as one persistent task communicating via `asyncio.Queue`.

**Settings cache**: `get_settings()` is `@cache`d and called at module level in `main.py`. Tests must call `get_settings.cache_clear()` after patching env vars — the autouse `clear_settings_cache` fixture in `tests/conftest.py` does this automatically.

**Mock tools are double-guarded**: registered only when `app_env != "production"` AND `enable_mock_tools` is set; import of `app/agents/tools_mock.py` is deferred. Never enable in production config.

**RAG sufficiency is structured, not parsed**: `_eval_agent`'s output type is `RelevanceVerdict` (`sufficient: bool` + non-empty `rationale`) — never text-match the reply. Nested retry budgets: pydantic-ai `retries={"output": ...}` for validation failures within one call, `_run_agent_with_retry` for transient failures across calls; a timeout returns the safe `sufficient=False` fallback with no retry.

**RAG workflow cache**: keyed by `sha256(query|max_retries|vector_store.generation)` — the generation counter invalidates pre-ingest entries without disturbing in-flight requests; thundering-herd protection via `_pending_futures`; `get_rag_workflow` caches per vector-store using a `WeakKeyDictionary`. The autouse `clear_workflow_cache` fixture resets that cache before/after every test.

**RAG uses the shared model chain**: `get_rag_workflow` reads `app.state.settings` and `app.state.llm_model` — the same `FallbackModel` chain the chat agent uses — instead of rebuilding a model. Either singleton absent fails the request with a flat-envelope 503 (`code="DEPENDENCY_NOT_INITIALIZED"`), never a silent fallback to process-global settings.

**Session ownership without a lookup table**: session ids are `{principal.id}.{token}.{signature}` (HMAC-signed). `authorize_session()` verifies with `secrets.compare_digest` and 403s on malformed/foreign ids. `POST /v1/agent/stream` can only *authorize* an existing id — no new id can be returned on that endpoint.

**Session trimming**: `trim_history()` (`app/stores/session_store/_trim.py`) is a pure function shared by both `SessionStore` backends, bounded by `session_max_messages` (default `1000`). Cuts land only between messages, never orphan a retained tool-call pair, and always keep `messages[0]` (the system prompt) — which is why the result can be `max_messages + 1` long, not exactly `max_messages`.

**Pluggable stores**: implement `typing.Protocol` in `app/stores/*/protocol.py`; register in `app/stores/factory.py`; wire via `lifespan`. Never subclass a concrete backend.

## Testing

- **Unit tests** (`tests/unit/`) have real network blocked via `tests/support/hermetic.py` — any missed mock raises `NetworkBlockedError` immediately
- **LLM substitute**: use `FunctionModel` (with `supports_json_schema_output=False` profile) from `tests/conftest.py::test_model` fixture — never use `TestModel` in integration/e2e layers
- **`build_test_settings(**overrides)`** helper in `conftest.py` constructs isolated `Settings` without relying on env vars
- **`asyncio_mode = "auto"`**: all coroutine tests are collected and awaited automatically — no `@pytest.mark.asyncio` needed
- **`EXPECT_LIVE_TESTS=N`** env var: session fails if the actual executed test count ≠ N — used to guard gated test lanes from silent skips
- **`chroma` marker**: opt-in (`RUN_CHROMA_INTEGRATION_TESTS` env var); downloads a Hugging Face embedding model — do not run routinely: `RUN_CHROMA_INTEGRATION_TESTS=1 EXPECT_LIVE_TESTS=6 uv run pytest tests/integration/test_chroma_query_with_scores.py` (`6` = `tests/support/chroma.py::CHROMA_LIVE_TEST_COUNT`)
- **`redis` marker**: gated on live reachability, not an opt-in var — `tests/support/redis.py::redis_reachable()` probes a real server and the lane skips (never fails) when none answers: `EXPECT_LIVE_TESTS=5 mise run test:redis` (`5` = `tests/support/redis.py::REDIS_LIVE_TEST_COUNT`)
- Tests for cloud-provider API key validation must `monkeypatch.delenv("LLM_API_KEY")` explicitly (the autouse fixture sets it by default)

### Repo-guard tests (assert on project structure, not app behavior)

These fail on structural drift — understand what they assert before bypassing:

- `test_file_size_policy.py` — hard-fails any `app/**`/`tests/**` file ≥1000 lines; warns on 500–999
- `test_contract_drift.py` — README's normative fences must not show classes/fields that no longer exist
- `test_ci_workflows.py` — every `uses:` in GitHub Actions YAML must be pinned to a 40-char SHA
- `test_pydantic_ai_api_lock.py` — subset-only lock on the pydantic-ai symbols/params/fields/kinds this project uses (no `app/` imports), so a dependency upgrade fails here by name. Add newly relied-upon symbols to its tables.
- `test_config_dependency_bounds.py` — every production dependency must declare an upper bound
- `test_no_hardcoded_model_ids.py`, `test_naming_conventions.py`, `test_pre_push_hook.py`, `test_expect_live_tests_plugin.py`, `test_block_network.py`, `test_pytest_config.py`, plus the `chroma`/`local`/`docker`/`redis` gating guards

## Dependency Pins That Are Load-Bearing

`fastapi<0.137` and `starlette<1.0` in `pyproject.toml` both exist because the newer version **silently disables the global rate limit** (router flattening / slowapi incompatibility) — read the comments there first; `tests/e2e/test_rate_limiting_enforcement.py` is the canary. The starlette pin is also why `mise run audit` carries `--ignore-vuln` entries: each is app-layer-mitigated or unreachable here. Re-run `uv run pip-audit` without them after any starlette bump.

`pydantic-ai-litellm` is pinned `<1.0` in `pyproject.toml` — that pin **should be `<0.3.0`** (it's a 0.x package and depends on six private pydantic-ai APIs; `<1.0` admits 0.3.x–0.9.x unreviewed). The correct bound mirrors how `fastapi` is handled for 0.x versioning.

A private-API coupling, not a version bound: the rate-limit-exceeded handler (`app/middleware/rate_limit.py`) delegates 429 header construction (`X-RateLimit-*`, delay-seconds `Retry-After`) to slowapi's `Limiter._inject_headers` — a leading-underscore method with no compatibility guarantee. A slowapi upgrade needs re-verification against `tests/unit/test_middleware_rate_limit_global_envelope.py`.

## CI Gating

- **PR CI** (`.github/workflows/pr.yml`): lint → `test:ci` → `test:redis` → `audit`. Never runs live LLM or Ollama tests, nor the `chroma`-marked tests (they self-skip unless `RUN_CHROMA_INTEGRATION_TESTS` is set, since they download a Hugging Face embedding model). The `redis`-marked lane does run here — CI starts a `redis:7-alpine` service container and sets `EXPECT_LIVE_TESTS=5` so a broken container fails the run instead of passing as a silent zero-collected green. `asyncio_mode = "auto"` means an unmarked coroutine test in this lane still runs instead of silently passing unawaited.
- **Nightly** (`.github/workflows/security.yml`): `pip-audit` + gitleaks, cron `37 3 * * *`.
- **pre-commit** (`.pre-commit-config.yaml`): gitleaks, `pip-audit`, `no-hardcoded-model-id` pygrep, `real-tool-conventions-guard` (fires on any non-mock `@agent.tool` under `app/agents/`, forcing review of `docs/tool-design-conventions.md`).
- **pre-push** (`.githooks/pre-push`, opt-in via `git config core.hooksPath .githooks`): probes Ollama, runs `EXPECT_LIVE_TESTS=5 mise run test:local` + `evals` when reachable (the pinned count guards against a lane that silently collects zero live cases), warns and lets the push through when not.
- **Dependabot** (`.github/dependabot.yml`): weekly `uv` + `github-actions`, minors/patches grouped. Its `ignore:` list blocks the forbidden bumps — `starlette` majors and `fastapi` **minors and majors** (Dependabot reads fastapi's 0.x releases by patch position, so `0.136 → 0.137` is a minor) — plus `chromadb`/`redis` majors, shelved pending a client-compatibility pass. Guarded by `tests/unit/test_dependabot_config.py`; `docs/dependency-runbook.md` is the accept/shelve process.

## Feature Status (`004-pydantic-ai-v2-unblock` active; `003-pydantic-ai-v2-migration` sealed, both tracked under `.sdd/specs/`)

- No known active bugs. The `InMemorySessionStore` LRU-victim defect previously listed here is fixed — victim selection now comes from `self._store.keys()` (`app/stores/session_store/in_memory.py`, which carries the comment explaining why).
- `003` shipped units 1–9; its task 9 recorded the adapter-compatibility gate **FAILED**, so its tasks 10–12 (Requirements 9–11) are closed as superseded by `004` rather than completed. `004` is the single active spec and re-executed that gate under its own Requirement 4 (run 1), recording it **PASSED** (evidence: `docs/adapter-probe-report-2026-08-13-run1.md`, cross-referencing the original 2026-08-11 finding at `docs/adapter-probe-report.md`), unblocking the v2 code migration, its behavioural pinning, and the Redis key-prefix cutover.
- `pydantic-ai-slim` stays pinned to the 1.x line (`>=1.99.0,<2.0` in `pyproject.toml`) until `004`'s gate is recorded as passed; only `004`'s task 7 retires this pin.

## Adding a New Real Agent Tool

Adding a non-mock `@agent.tool` under `app/agents/` triggers the `real-tool-conventions-guard` pre-commit hook. Read `docs/tool-design-conventions.md` before committing.
