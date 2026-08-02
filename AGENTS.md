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
mise run test:benchmark      # latency/throughput/cache-hit benchmarks (-s)
mise run test:local          # requires a running Ollama instance (-m ollama)
mise run lint                # ruff check + ty check (type checker is `ty`, NOT mypy)
mise run format              # ruff format
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

## Architecture

`app/main.py` is the composition root — `create_app()` (factory) is used everywhere; `app = create_app()` at module bottom is the uvicorn entrypoint. Tests always call `create_app(settings=..., model=test_model)` directly, never import the module-level `app` singleton.

**Middleware order is inverted** — FastAPI executes in *reverse* registration order. `TrustedHostMiddleware` is added last so it runs first (ahead of CORS); SecurityHeaders added first so it runs last. Do not reorder without reading the comments in `main.py`.

**Host validation** — `allowed_hosts` defaults to `["*"]` for dev, but `Settings` rejects `"*"` whenever `app_env != "development"` (staging and production both): starlette rebuilds redirect `Location` from the `Host` header and `redirect_slashes` is on, so a wildcard allow-list is an unauthenticated open-redirect. Wildcards use starlette's `*.example.com` form, not Django's `.example.com`.

**LLM model format**: `"provider:model"` (e.g. `openai:gpt-4o`) — converted to LiteLLM's `"provider/model"` internally by `build_model()` in `app/agents/chat_agent.py`.

**Ollama URL gotcha**: LiteLLM auto-appends `/v1`, so the chat model uses `http://localhost:11434` (no `/v1`). `OllamaEmbeddingVectorStore` calls Ollama directly and needs `http://localhost:11434/v1`. Do not "fix" this asymmetry.

**Settings cache**: `get_settings()` is `@cache`d and called at module level in `main.py`. Tests must call `get_settings.cache_clear()` after patching env vars — the autouse `clear_settings_cache` fixture in `tests/conftest.py` does this automatically.

**Mock tools are double-guarded**: registered only when `app_env != "production"` AND `enable_mock_tools` is set; import of `app/agents/tools_mock.py` is deferred. Never enable in production config.

**RAG workflow cache**: keyed by `sha256(query|max_retries)`; thundering-herd protection via `_pending_futures`; `get_rag_workflow` caches per vector-store using a `WeakKeyDictionary`. The autouse `clear_workflow_cache` fixture resets both caches before/after every test.

## Testing

- **Unit tests** (`tests/unit/`) have real network blocked via `tests/support/hermetic.py` — any missed mock raises `NetworkBlockedError` immediately
- **LLM substitute**: use `FunctionModel` (with `supports_json_schema_output=False` profile) from `tests/conftest.py::test_model` fixture — never use `TestModel` in integration/e2e layers
- **`build_test_settings(**overrides)`** helper in `conftest.py` constructs isolated `Settings` without relying on env vars
- **`EXPECT_LIVE_TESTS=N`** env var: session fails if the actual executed test count ≠ N — used to guard gated test lanes from silent skips
- Tests for cloud-provider API key validation must `monkeypatch.delenv("LLM_API_KEY")` explicitly (the autouse fixture sets it by default)
