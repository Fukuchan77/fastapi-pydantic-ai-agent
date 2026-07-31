# OWASP Agentic AI / LLM Top 10 Mapping

Satisfies Req 16.1/16.2 (`.sdd/specs/001-agent-architecture-enhancements/spec.md`).
Maps each risk in the **OWASP Top 10 for LLM Applications (2025)** — the
canonical "LLM Top 10", which folds agentic-autonomy risk into LLM06 Excessive
Agency — to this project's concrete mitigating control, or to an explicit
acceptance/deferral where no dedicated control exists yet. Every row cites the
implementing module and, where one exists, its test file, so a reviewer can
verify the claim against running code rather than prose.

This document owns the mapping only; each control is owned and maintained by
its own component (see `plan.md`'s Components section). Update this table
whenever a cited file's mitigating behavior changes materially, not on every
unrelated edit to that file.

| # | Risk | Status | Mitigating control | Citation |
|---|------|--------|---------------------|----------|
| LLM01 | Prompt Injection | Partial / accepted | No dedicated input-side prompt-injection filter exists. Blast radius is bounded instead: even a successful injection cannot escape the tool allow-list, approval hook, or token budget. | `app/agents/guardrails.py` (`run_guarded`, `_GuardedToolset.call_tool`); `tests/unit/agents/test_guardrails.py` |
| LLM02 | Sensitive Information Disclosure | Mitigated | Logfire scrubbing of `prompt`/`tool_input`/`tool_output` is enabled by default; disabling it (`log_sensitive_payloads=True`) emits an `AUDIT:` warning. SSE/tool-call payload models use `extra="forbid"` so an unlisted field cannot silently leak. `Principal` identity is a one-way hash of the API key — the key itself is never stored or logged. | `app/observability.py` (`configure_logfire`); `app/config.py` (`log_sensitive_payloads`); `app/patterns/sse.py` (`_StrictEvent`); `app/security/principal.py` (`derive_principal_id`); `tests/unit/test_observability.py` |
| LLM03 | Supply Chain | Mitigated | `pip-audit` runs on every PR and nightly on a cron; a local `pip-audit` pre-commit hook and `gitleaks` block a commit that introduces a vulnerable dependency or a secret; Dependabot keeps the `uv` dependency tree and pinned GitHub Actions current weekly; all third-party Actions are pinned to a commit SHA. | `.github/workflows/pr.yml`; `.github/workflows/security.yml`; `mise.toml` (`tasks.audit`); `.pre-commit-config.yaml`; `.github/dependabot.yml` |
| LLM04 | Data and Model Poisoning | Partial / accepted | `POST /v1/rag/ingest` requires a valid API key, limiting who can write into the vector store. No content sanitization or anomaly scoring is applied to ingested text — deferred; this project does no model training/fine-tuning, so poisoning risk is scoped to the RAG corpus only. | `app/api/v1/rag.py` (`ingest`, `Depends(verify_api_key)`) |
| LLM05 | Improper Output Handling | Mitigated | Citations returned to the caller are workflow-constructed from the deterministically-ordered, truncated hits actually fed to the synthesis prompt (never model-asserted), so a citation can't reference a chunk outside the current run. `validate_citations`/`order_hits` enforce this as defense-in-depth; `EmptyCitationError`/`DanglingCitationError` map to HTTP 502 with a generic message (no id/content leakage). At the retry limit the workflow returns a degraded answer grounded only in the hits it actually has, never a free-form fallback. | `app/workflows/citation.py`; `app/workflows/corrective_rag.py` (`search`/`evaluate`/`synthesize`); `app/api/v1/rag.py` (exception mapping); `tests/unit/workflows/test_citation.py` |
| LLM06 | Excessive Agency | Mitigated | `run_guarded()` wraps every tool-capable agent run with a native `toolset.filtered` allow-list, an `approval_hook`, a pre-side-effect token-budget check, and a closed `StopReason` vocabulary (`disallowed_tool`/`denied`/`budget_exceeded`/`max_iterations`/`completed`) plus an audit trail — applied identically on both the non-streaming and SSE-streaming chat paths. Session ownership is bound to the calling `Principal`; `authorize_session` returns 403 on any cross-principal session-id reuse. | `app/agents/guardrails.py`; `app/security/principal.py`; `app/services/session_service.py`; `tests/unit/agents/test_guardrails.py`; `tests/unit/services/test_session_service.py` |
| LLM07 | System Prompt Leakage | Accepted | The system prompt contains no secrets or credentials by construction (session-signing keys, API keys, etc. are never interpolated into it). The `NativeOutput(ChatOutput)` structured-output schema (where the active model profile supports it) constrains the response shape, reducing incidental verbatim disclosure. If a leak did occur, LLM02's Logfire scrubbing still limits its blast radius in logs/traces. No dedicated leak-detection test exists — accepted given the prompt carries no sensitive content today. | `app/agents/chat_agent.py` (`_build_system_prompt`, `ChatOutput`) |
| LLM08 | Vector and Embedding Weaknesses | Partial / accepted | Ingestion requires authentication (same control as LLM04), limiting who can influence retrieval ranking. `query_with_scores()` exposes a per-hit relevance score and a stable `chunk_id` (`{source}::{ordinal:04d}`) for every retrieved hit, giving a post-hoc audit trail back to the ingestion batch that produced it. No defense against embedding-inversion or adversarial-perturbation attacks is implemented — deferred. | `app/stores/vector_store.py` (`query_with_scores`); `app/models/rag.py` (`RetrievedHit`) |
| LLM09 | Misinformation | Partial / mitigated | The CRAG workflow widens retrieval (`rag_initial_k` → `rag_widened_k`) and retries evaluation before synthesizing, and terminates early with a canned response on zero hits rather than generating from nothing; every synthesized claim is grounded to a specific retrieved chunk id (see LLM05). The offline two-axis (Outcome/Behavior) LLM-judge eval suite against a golden dataset provides a regression signal for factuality-adjacent behavior, gated on `mise run evals` pre-push. | `app/workflows/corrective_rag.py`; `evals/graders.py`; `evals/runner.py`; `evals/golden/basic_qa.json` |
| LLM10 | Unbounded Consumption | Mitigated | SSE streams are capped at `sse_max_events`, each send has a `sse_send_timeout`, and idle periods emit a `sse_heartbeat_interval` heartbeat. `POST /v1/agent/chat` is wrapped in `asyncio.wait_for(chat_request_timeout)`. Native `UsageLimits` (`usage_request_limit`/`usage_total_tokens_limit`) plus a pre-tool-call budget re-check in `run_guarded` bound token/request usage per run. Every LLM-invoking route (`/v1/agent/chat`, `/v1/agent/stream`, `/v1/rag/query`) carries a stricter `llm_rate_limit` (default `30/minute`) than the global default, backed by Redis with an in-memory fallback. The RAG workflow itself has a `rag_workflow_timeout` ceiling independent of the per-LLM-call `llm_agent_timeout`. | `app/patterns/sse.py`; `app/api/v1/_stream.py`; `app/agents/guardrails.py`; `app/middleware/rate_limit.py` (`enforce_llm_rate_limit`); `app/config.py` (`sse_max_events`, `sse_send_timeout`, `sse_heartbeat_interval`, `chat_request_timeout`, `usage_request_limit`, `usage_total_tokens_limit`, `llm_rate_limit`, `rag_workflow_timeout`, `llm_agent_timeout`); `tests/unit/api/v1/test_stream_lifecycle.py`; `tests/e2e/test_rate_limiting_enforcement.py` |

## Notes on scope

- **"Mitigated"** means a concrete, tested control exists and is wired into
  the production path (not just a dev-only or conditional path).
- **"Partial / accepted"** means a real control reduces the risk's blast
  radius or likelihood but does not eliminate it, and the residual risk is
  knowingly accepted rather than silently unaddressed.
- **"Accepted"** means the risk is judged low today given the current design
  (e.g. no secrets in the system prompt) with no dedicated control — revisit
  if that assumption changes (e.g. the system prompt starts carrying
  sensitive context).
- Requirement 15's tool-design conventions (naming, pagination, lenient
  parsing) are a code-quality/reliability concern for future real tools, not
  a distinct OWASP Top 10 risk, and are tracked separately in
  `docs/tool-design-conventions.md`.
