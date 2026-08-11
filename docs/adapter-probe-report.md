# Adapter-compatibility gate report (Requirement 8, `003-pydantic-ai-v2-migration` unit 9)

Executed: 2026-08-11. This is the evidence artifact Requirement 8.6 requires: the single
branch point of `003-pydantic-ai-v2-migration` — whether `pydantic-ai-litellm` works against
`pydantic-ai-slim` 2.x — determined by execution rather than by reading changelogs. This
document changes no application code; it records what was run, in a disposable environment,
and what happened.

**Verdict: FAILED** (Requirement 8.5 — a third, unpredicted failure occurred). Per Requirement
8.8, Requirements 9, 10, and 11 are recorded as **unmet**; the alternatives (patch the adapter,
vendor it, or replace the LiteLLM routing with pydantic-ai's native OpenAI-compatible route) are
deferred to a separate follow-up spec rather than pursued in `003-pydantic-ai-v2-migration`.

The verdict is mechanical, per 8.4/8.5's letter, and is not softened by the fact that the third
failure turns out to be low-severity (§5) or that a one-line supplementary fix (§6) shows the
adapter's actual request path works fine against a real provider under 2.x. The gate's rule
does not have a severity threshold; it has a count.

## 1. Isolation mechanism (Requirement 8.1, 8.2; ADR-6)

- **Mechanism**: a separate `git worktree`, added from the repository root while on
  `003-pydantic-ai-v2-migration` at commit `eef3534`:
  ```
  git worktree add ./.worktrees/adapter-probe -b probe/adapter-pydantic-ai-2.27
  ```
  The new branch (`probe/adapter-pydantic-ai-2.27`) exists only to give the worktree a ref; it
  carries no commits of its own and none were made in it.
- **Dedicated virtual environment**: the worktree's own `.venv` at
  `.worktrees/adapter-probe/.venv`, created and used exclusively via `uv`'s `--project
  .worktrees/adapter-probe` flag (so every `uv`/`pytest` invocation below ran against that
  environment, never the repository's own `.venv`), pinned explicitly to `--python 3.13` — see
  §4 for why the interpreter version had to be pinned rather than left to `uv`'s default choice.
- **Constraint edit, worktree-only**: `pyproject.toml`'s `pydantic-ai-slim[logfire,openai]`
  bound was changed from `>=1.99.0,<2.0` to `>=2.27.0,<3.0` **inside the worktree only**, with a
  comment recording why. `pydantic-ai-litellm`'s bound (`>=0.2.3,<1.0`) was left untouched —
  its current release (0.2.8) already satisfies it.
- **Lock refresh, scope-preserving**: `uv lock --project .worktrees/adapter-probe
  --upgrade-package pydantic-ai-slim --upgrade-package pydantic-ai-litellm` — the same
  constraint-preserving primitive Task 6 used, deliberately *not* a bare `uv lock` (a first,
  discarded attempt at a full re-lock is recorded as a methodology note in §4, because it
  silently admits unrelated collateral version drift into the signal). The resulting diff to
  `uv.lock`, confirmed with `git diff -- uv.lock` from inside the worktree, touches **only** the
  `pydantic-ai-slim` and `pydantic-graph` (its forced companion) version fields and the new
  `anyio` dependency edge they bring; every other resolved package is byte-identical to the
  repository's current lock.
- **No commit was made from inside the worktree at any point.** `git status --short` from
  inside the worktree at the end of the session:
  ```
   M pyproject.toml
   M uv.lock
  ?? pip_freeze.txt
  ```
  (`pip_freeze.txt` and the per-stage `*.log` files are this probe's own uncommitted scratch
  output, harmless and never staged.)
- **The repository's own environment and lock file are unmodified.** `git status --short` from
  the repository root at the end of the session:
  ```
  ?? .worktrees/
  ```
  No tracked file changed. The repository's `.venv` was never invoked by any command in this
  report — every `uv`/`pytest` call below is explicit about `--project .worktrees/adapter-probe`.

## 2. Complete resolved version set (Requirement 8.6)

Captured via `uv run --project .worktrees/adapter-probe python -m pip list --format=freeze`
inside the worktree's own venv (not `uv pip list`, which — a mistake caught and corrected during
this run — resolves against the ambient/default project rather than an arbitrary `--project`
target and would have silently reported the repository's own versions instead of the worktree's).

Interpreter: **CPython 3.13.7** (pinned; see §4).

The two packages this gate exists to test, and their forced companion:

| Package | Repository's current lock | This gate's resolution |
|---|---|---|
| `pydantic-ai-slim` | 1.107.2 | **2.27.1** |
| `pydantic-graph` (forced companion) | 1.107.2 | **2.27.1** |
| `pydantic-ai-litellm` | 0.2.8 | 0.2.8 (unchanged — already its latest release) |
| `litellm` (adapter's own dependency) | 1.94.1 | 1.94.1 (unchanged) |

Every other package resolved identically to the repository's current `uv.lock` — confirmed by
the `git diff -- uv.lock` scope in §1, and spot-checked in the full freeze below against values
the project's own recent `pdca/do.md` entries recorded (`fastapi` 0.136.3, `starlette` 0.52.1,
`slowapi` 0.1.10, `chromadb` 0.6.3, `redis` 5.3.1, `sentence-transformers` 5.7.0, `ruff` 0.16.2,
`ty` 0.0.70 — all match).

Full freeze (184 packages):

```
aiohappyeyeballs==2.7.1
aiohttp==3.14.3
aiosignal==1.4.0
aiosqlite==0.22.1
annotated-doc==0.0.5
annotated-types==0.8.0
anyio==4.14.2
asgiref==3.12.1
attrs==26.1.0
backoff==2.2.1
banks==2.4.5
bcrypt==5.0.0
boolean.py==5.0
build==1.5.0
CacheControl==0.14.4
certifi==2026.7.22
cfgv==3.5.0
charset-normalizer==3.4.9
chroma-hnswlib==0.7.6
chromadb==0.6.3
click==8.4.2
colorama==0.4.6
coverage==7.15.2
cyclonedx-python-lib==11.11.0
dataclasses-json==0.6.7
defusedxml==0.7.1
Deprecated==1.3.1
dirtyjson==1.0.8
distlib==0.4.3
distro==1.9.0
durationpy==0.10
executing==2.2.1
fastapi==0.136.3
fastuuid==0.14.0
filelock==3.32.2
filetype==1.2.0
flatbuffers==25.12.19
frozenlist==1.8.0
fsspec==2026.7.0
genai-prices==0.1.1
googleapis-common-protos==1.75.0
greenlet==3.5.4
griffe==2.1.0
griffecli==2.1.0
griffelib==2.1.0
grpcio==1.83.0
h11==0.16.0
hf-xet==1.5.2
httpcore==1.0.9
httpcore2==2.9.1
httptools==0.8.0
httpx==0.28.1
httpx2==2.9.1
huggingface_hub==1.26.0
identify==2.6.19
idna==3.18
importlib_metadata==8.9.0
importlib_resources==7.1.0
iniconfig==2.3.0
Jinja2==3.1.6
jiter==0.16.0
joblib==1.5.3
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
kubernetes==36.0.3
license-expression==30.4.4
limits==5.8.0
litellm==1.94.1
llama-index-core==0.14.23
llama-index-instrumentation==0.5.0
llama-index-workflows==2.23.0
logfire==4.39.0
logfire-api==4.39.0
markdown-it-py==4.2.0
MarkupSafe==3.0.3
marshmallow==3.26.2
mdurl==0.1.2
mmh3==5.2.1
mpmath==1.3.0
msgpack==1.2.1
multidict==6.7.1
mypy_extensions==1.1.0
narwhals==2.24.0
nest-asyncio==1.6.0
networkx==3.6.1
nltk==3.10.1
nodeenv==1.10.0
numpy==2.5.1
oauthlib==3.3.1
onnxruntime==1.28.0
openai==2.52.0
opentelemetry-api==1.44.0
opentelemetry-exporter-otlp-proto-common==1.44.0
opentelemetry-exporter-otlp-proto-grpc==1.44.0
opentelemetry-exporter-otlp-proto-http==1.44.0
opentelemetry-instrumentation==0.65b0
opentelemetry-instrumentation-asgi==0.65b0
opentelemetry-instrumentation-fastapi==0.65b0
opentelemetry-instrumentation-httpx==0.65b0
opentelemetry-proto==1.44.0
opentelemetry-sdk==1.44.0
opentelemetry-semantic-conventions==0.65b0
opentelemetry-util-http==0.65b0
orjson==3.11.9
overrides==7.7.0
packageurl-python==0.17.6
packaging==26.2
pillow==12.3.0
pip==26.2
pip-api==0.0.34
pip_audit==2.10.1
pip-requirements-parser==32.0.1
platformdirs==4.11.0
pluggy==1.6.0
posthog==7.35.4
pre_commit==4.6.1
propcache==0.5.2
protobuf==7.35.1
py-serializable==2.1.0
pydantic==2.13.4
pydantic-ai-litellm==0.2.8
pydantic-ai-slim==2.27.1
pydantic_core==2.46.4
pydantic-graph==2.27.1
pydantic-settings==2.15.0
Pygments==2.20.0
PyJWT==2.13.0
pyparsing==3.3.2
PyPika==0.51.1
pyproject_hooks==1.2.0
pytest==9.1.1
pytest-asyncio==1.4.0
pytest-cov==7.1.0
python-dateutil==2.9.0.post0
python-discovery==1.5.1
python-dotenv==1.2.2
PyYAML==6.0.3
redis==5.3.1
referencing==0.37.0
regex==2026.7.19
requests==2.34.2
requests-oauthlib==2.0.0
rich==15.0.0
rpds-py==2026.6.3
ruff==0.16.2
safetensors==0.8.0
scikit-learn==1.9.0
scipy==1.18.0
sentence-transformers==5.7.0
setuptools==83.0.0
shellingham==1.5.4
six==1.17.0
slowapi==0.1.10
sniffio==1.3.1
sortedcontainers==2.4.0
SQLAlchemy==2.0.51
starlette==0.52.1
sympy==1.14.0
tenacity==9.1.4
threadpoolctl==3.6.0
tiktoken==0.13.0
tinytag==2.3.0
tokenizers==0.22.2
tomli==2.4.1
tomli_w==1.2.0
torch==2.13.0
tqdm==4.70.0
transformers==5.14.1
truststore==0.10.4
ty==0.0.70
typer==0.27.0
typing_extensions==4.16.0
typing-inspect==0.9.0
typing-inspection==0.4.2
urllib3==2.7.0
uvicorn==0.52.1
uvloop==0.22.1
virtualenv==21.7.1
watchfiles==1.2.0
websocket-client==1.9.0
websockets==17.0.1
wrapt==2.3.0
yarl==1.24.5
zipp==4.1.0
```

## 3. Six-stage execution and outcomes (Requirement 8.3)

Run in order, all against the environment in §1/§2. Numbers below are the corrected,
confound-free measurement (Python 3.13.7) — see §4 for why an initial run under a different
interpreter is not used as the gate's measurement.

| # | Stage | Command | Result |
|---|---|---|---|
| 1 | pydantic-ai API lock test | `pytest tests/unit/test_pydantic_ai_api_lock.py -v` | **1 failed, 23 passed** |
| 2 | Unit tier | `pytest tests/unit/ -q` | **50 failed, 1255 passed, 1 skipped** |
| 3 | Integration tier | `pytest tests/integration/ -q` | **18 failed, 29 passed, 18 skipped** |
| 4 | E2E tier | `pytest tests/e2e/ -q` | **3 failed, 0 passed, 42 errors** |
| 5 | Ollama-gated local lane | `EXPECT_LIVE_TESTS=5 pytest tests/local/ -v -m ollama` | **5 failed, 0 passed** (all 5 pinned live cases reached `call`, none skipped — the anti-false-green count matches) |
| 6 | Evals run | `python -m evals.runner` | **crashed before grading any case** (exit 1, at agent construction) |

Skips in stages 2/3 are the pre-existing opt-in/reachability-gated lanes (Chroma
self-skips without `RUN_CHROMA_INTEGRATION_TESTS`; Docker self-skips without a daemon; Redis
self-skips — unreachable in this environment) and are unrelated to this gate; they skip
identically against the repository's current (1.x) resolution.

**Stage 5's expected-live-test count is 5** — the same value `tests/support/ollama.py` implies
and `.githooks/pre-push` pins (`EXPECT_LIVE_TESTS=5 mise run test:local`). Recording this here,
at the time this gate report is authored, is how Requirement 8.3's "the two SHALL be updated
together" is discharged for this gate specifically: **this gate was executed at the pre-bump
count of 5.** Task 10.5 (unit 10, still open as of this report) is free to raise that count in
the repository's own test suite; doing so does not retroactively change what this unrepeatable
gate run measured, and this statement is what keeps the two from silently drifting apart.

## 4. Methodology correction: an interpreter-version confound, caught and corrected

The worktree's `.venv` was first created without pinning an interpreter (`uv sync --project
.worktrees/adapter-probe`, no `--python` flag). `uv` selected the newest interpreter satisfying
`requires-python = ">=3.13"` that it had available: **CPython 3.14.5** — not the CPython 3.13.7
the repository's own `.venv` actually uses. That is a second variable changed at the same time as
the pydantic-ai bump, and it produced real, but irrelevant, gate noise:

- Running the unit tier under 3.14.5 first produced **91 failed, 12 errors** (vs. the corrected
  50 failed above) and, separately, a genuine multi-minute **hang** in
  `tests/integration/test_rag_cache_generation.py::test_pending_query_survives_generation_bump_during_ingest`
  that the test's own internal `asyncio.wait_for(..., timeout=5)` guard did not catch (the hang
  occurred at an *earlier*, unguarded `await eval_started.wait()`, before the guarded section is
  reached).
- Root cause, isolated by re-running the exact same stage with only the interpreter pinned back
  to 3.13: Python 3.14 deprecates `asyncio.iscoroutinefunction` (removal slated for 3.16), and
  `llama-index-workflows`'s own runtime (`workflows/runtime/types/step_function.py`) still calls
  it. This project's own `error::DeprecationWarning` pytest filter (`pyproject.toml`, Requirement
  7) is deliberately bare/unqualified (ADR-5 — a module-keyed filter cannot fire for a warning
  attributed to a third-party caller), so it turns that stdlib deprecation into a hard test
  failure — and, in one specific test's timing, into the observed hang. **This is a real
  instance of the exact failure mode plan.md's C9 "Class B" classification anticipated**
  (*"a warning that 2.x, or a transitively newer dependency, triggers... would surface as a hard
  error and be indistinguishable from a real incompatibility"*) — except the trigger here was a
  transitively newer **interpreter**, not a transitively newer dependency, which the plan's prose
  did not name but its underlying mechanism covers identically.
- Once the venv was rebuilt with `--python 3.13` (matching the repository's own environment), the
  unit tier's result dropped to exactly the 50 failures in §3/§5 with **zero** `DeprecationWarning`
  occurrences anywhere across all five pytest stages (`grep -c DeprecationWarning` on every stage
  log: 0), and the previously-hanging integration test passed outright in 0.15s. No Class-B
  re-run with `-W default::DeprecationWarning` was needed for the final measurement, because
  pinning the interpreter removed the confound at its source rather than papering over it at the
  filter.

**The corrected, interpreter-matched run (Python 3.13.7) is the gate's measurement.** The 3.14.5
run is recorded here only so a future re-run of this gate does not silently reintroduce the same
confound: pin `--python` to whatever the repository's own `.venv` resolves before drawing any
conclusion from a failure count.

A second, smaller methodology correction, also recorded so it is not repeated: the first lock
refresh used a bare `uv lock --project .worktrees/adapter-probe` (a full re-resolve), which
picked up whatever the newest-compatible release of *every* package happened to be at run time —
collateral drift unrelated to pydantic-ai that would have contaminated the failure signal with
noise from other packages' own newer releases. This was caught before running any test stage
(`git diff -- uv.lock` showed changes beyond the two pydantic-ai packages) and corrected by
resetting `uv.lock` (`git checkout -- uv.lock` inside the worktree) and re-locking with the
scope-preserving `--upgrade-package pydantic-ai-slim --upgrade-package pydantic-ai-litellm` form
instead (§1).

## 5. Failure classification (Requirement 8.3's classification step; plan.md C9)

Every failure across all six stages was read to its root cause. `grep`-ing every stage's captured
`^E   `/`^E       ` assertion lines and excluding the two predicted signatures below turns up
**exactly one** residual, distinct failure signature — in the unit tier only. No stage produced
any Class-B (deprecation-filter-induced) failure once §4's confound was removed, so no stage
needed the `-W default::DeprecationWarning` re-run 8.3/C9 describes; the classification below is
therefore a plain tally of root causes, not a filter-relaxation exercise.

### 5.1 Predicted failure 1 — the model-profile canary (Requirement 8.4, first predicted failure)

- **Site**: `tests/unit/test_pydantic_ai_api_lock.py::TestDataclassFieldAndAnnotationSubsets::test_model_profile_has_the_field_the_known_v2_break_lands_on`
- **Verbatim failure**:
  ```
  AttributeError: type object 'ModelProfile' has no attribute '__dataclass_fields__'
  ```
- **Cause**: `pydantic_ai.profiles.ModelProfile` changes from a dataclass to a `TypedDict` in
  2.x (upstream PR #5481, per `research.md`/`spec.md`), so `__dataclass_fields__` no longer
  exists on the type at all. This is the deliberately planted canary — it fires exactly as
  designed, in exactly the one place it was placed to fire.
- **Count**: 1 test (stages 1 and 2 — it is the same test, run once directly and once as part of
  the full unit tier).

### 5.2 Predicted failure 2 — the native-output capability read (Requirement 8.4, second predicted failure)

- **Site**: `app/llm/factory.py:91`, inside `supports_native_output()`:
  ```python
  target = model.models[0] if isinstance(model, FallbackModel) else model
  return target.profile.supports_json_schema_output
  ```
- **Verbatim failure** (identical at every occurrence, across every stage):
  ```
  AttributeError: 'dict' object has no attribute 'supports_json_schema_output'
  ```
- **Cause**: the same `ModelProfile` shape change as §5.1 — an *instance* of the now-`TypedDict`
  profile is a plain `dict` at runtime, so attribute-style access raises. `app/agents/chat_agent.py`'s
  `build_chat_agent()` calls `supports_native_output(infer_model(resolved_model))`
  **unconditionally**, with no fallback and no gating on model type, so this single line blocks
  agent construction entirely — for every model, in every code path that builds the chat agent
  (which is every stage's app-startup/lifespan path, since none of them substitutes a
  pre-built agent that skips `build_chat_agent()`).
- **Count and manifestation, by stage** (one root cause, many symptoms — this is why the total
  failure count across stages is large despite there being only two *predicted* defects):
  | Stage | Failing/erroring tests | How it's reached |
  |---|---|---|
  | 1 (lock test) | 0 | not exercised by this stage |
  | 2 (unit) | 42 | any test whose fixture builds a real app/agent (`create_app`, `build_chat_agent`, or `supports_native_output`/`TestSupportsNativeOutput` directly) |
  | 3 (integration) | 18 | `test_lifespan*.py`, `test_agent_session.py`, `test_agent_stream_native_output.py`, `test_shared_chain_concurrency.py`, `test_store_dry_run_startup.py` — all drive real lifespan startup |
  | 4 (e2e) | 45 (3 failed + 42 errors) | every e2e test uses the `client`/app fixture, which runs full lifespan startup |
  | 5 (Ollama lane) | 5 | `build_chat_agent()` against a real Ollama-backed model — same crash, before any request is sent |
  | 6 (evals) | 1 (the whole run) | `evals/runner.py::_run_against_live_agent` calls `build_chat_agent(settings=settings)` directly |
  | **Total** | **111** | |

### 5.3 Unpredicted failure — the `UsageLimitExceeded` message template (Requirement 8.5's "third failure")

- **Site**: `tests/unit/agents/test_usage_limit_templates.py` (7 tests, all in
  `TestUsageLimitMessageTemplates`), each asserting `str(exc_info.value) ==` an exact,
  fully-specified template string for one of pydantic-ai's own
  `UsageLimits.check_before_request`/`check_before_tool_call`/`check_tokens` exception messages.
- **Verbatim failure** (one representative; all 7 have the identical shape — an appended
  sentence at the end of an otherwise-unchanged message):
  ```
  AssertionError: assert 'The next req...st_limit of 1' == 'The next req...#usage-limits'
    - The next request would exceed the request_limit of 1
    + The next request would exceed the request_limit of 1. Consider raising the limit, or see the docs on usage limits for budget-aware patterns: https://ai.pydantic.dev/agent/#usage-limits
  ```
- **Cause**: pydantic-ai 2.27.1's `UsageLimitExceeded.__str__` appends a help-text sentence
  ("Consider raising the limit, or see the docs on usage limits for budget-aware patterns:
  https://ai.pydantic.dev/agent/#usage-limits") to every one of these messages, which 1.107.2
  does not. This is an upstream wording change, not a shape or behavior change, and it is
  exactly the kind of thing the API lock test's own docstring already named as an unlockable
  blind spot: *"The `str(UsageLimitExceeded)` substring parsing in
  `app/agents/guardrails.py::classify_usage_limit_exceeded` has no importable symbol or kind to
  lock — it is behaviour, not shape — so it is coverable only behaviourally, by the pinned-template
  regression test... not by anything here."* That regression test is precisely what breaks.
- **Production impact — none.** `app/agents/guardrails.py::classify_usage_limit_exceeded()`
  reads the same message but only checks substring containment:
  ```python
  if "request_limit" in message or "tool_calls_limit" in message:
      return "max_iterations"
  return "budget_exceeded"
  ```
  Both substrings still occur verbatim inside the v2 message, unchanged; the appended sentence
  only follows them. `classify_usage_limit_exceeded()`'s actual `StopReason` mapping is
  unaffected by this upstream change — only the *test* that pins the exact full string breaks,
  because it over-specifies (equality) what the production code only needs (containment).
- **Count**: 7 tests, unit tier only (stage 2). Not observed in any other stage (this exception
  path is unit-tested directly against `UsageLimits`, not exercised through a live model call in
  integration/e2e/local/evals).
- **Why this is still recorded as the disqualifying third failure**: Requirement 8.4/8.5 does not
  carve out an exception for a failure whose production impact is nil — it counts *failures*,
  and a mechanical, severity-blind rule is the entire point of pre-committing to it before running
  the gate (so the result can't be argued into passing after the fact). §7 records the
  behavior-preservation finding as context for whoever scopes the follow-up spec Requirement 8.8
  defers to, not as a basis for reclassifying this as tolerated.

### 5.4 Tally against the pass/fail rule

- Distinct failure root causes found: **3** (§5.1, §5.2, §5.3).
- Predicted (Requirement 8.4): §5.1, §5.2 — exactly 2, exactly as named.
- Third failure (Requirement 8.5): §5.3.
- **Result: the gate is recorded as FAILED**, per 8.5's unconditional "any third failure" rule.

## 6. Supplementary diagnostic (not part of the graded gate; exploratory only)

Because §5.2's defect blocks `build_chat_agent()` unconditionally, none of the six graded stages
above ever got past agent *construction* far enough to exercise the `pydantic-ai-litellm`
adapter's actual request/streaming code against a real model under 2.x — the substantive question
Requirement 8's preamble names (*"the Ollama lane and the evals run are the only paths that drive
the adapter's streaming code against a real provider"*) was therefore **not actually answered** by
the graded run, only blocked before it could be.

To find out anyway, §5.2's fix — already planned as task 10.2, `.get("supports_json_schema_output",
False)` instead of attribute access — was applied **as a temporary, uncommitted edit inside the
disposable worktree only**, purely to see past the blocker. This is explicitly outside
Requirement 8's "no code" contract for the graded gate (§1's `git status` evidence and §3's stage
results above are all from *before* this edit); it is recorded separately, as exploratory
evidence, and the edit was reverted (`git checkout -- app/llm/factory.py` inside the worktree)
immediately after, before this report was finalized. **It does not change §5's verdict.**

With that one line patched:

- **The Ollama-gated local lane (stage 5) passed in full: 5/5**, in 133s, against real
  `ollama:granite4.1:8b` and `ollama:llama3.2:latest` models — basic completion, a mock-tool call
  loop terminating within its configured limits, and multi-turn session history, all driven
  through the actual `pydantic_ai_litellm.LiteLLMModel` adapter and its six private-API
  dependencies (`Model._get_instruction_parts`, `_utils.PeekableAsyncStream`/`Unset`,
  `_parts_manager.handle_text_delta`, `check_allow_model_requests`, `get_user_agent`,
  `StreamedResponse._usage` — the set `research.md`'s risk note names) against pydantic-ai-slim
  2.27.1. This is strong practical evidence that the adapter's core mechanism — not just its
  declared metadata — remains compatible with the 2.x line in the one area this gate cared about
  most.
- **The evals run (stage 6) still did not complete** — but for a reason unrelated to pydantic-ai
  2.x or the adapter: with no `JUDGE_MODEL` configured, the run self-grades with the same small
  local `ollama:llama3.2:latest` model, which returned its `Rating.score` as the string `'4'`
  instead of the integer `4`, failing pydantic validation after exhausting output retries
  (`pydantic_ai.exceptions.UnexpectedModelBehavior: Exceeded maximum output retries (2)`). This is
  a pre-existing reliability limitation of grading with a small local model with no independent
  judge configured (the run itself prints an `AUDIT:` warning about exactly this before
  attempting it), orthogonal to this gate's question, and is not evidence against adapter
  compatibility.

## 7. What this means for Requirements 9–11 (Requirement 8.8)

Per Requirement 8.8, because §5.4 records the gate as failed:

- **Requirements 9 (v2 code migration), 10 (v2 behavioural pinning), and 11 (cutover key prefix)
  are recorded as unmet.** No task under units 10–12 may proceed on the strength of this gate.
- **The failure-branch alternatives — patching the adapter, vendoring it, or replacing the
  LiteLLM routing with pydantic-ai's native OpenAI-compatible route — are deferred to a separate
  follow-up spec**, not pursued inside `003-pydantic-ai-v2-migration`.
- For whoever scopes that follow-up spec: §5.1 and §5.2 are both trivial, already-diagnosed,
  one-line-each fixes (§6 demonstrates §5.2's fix alone is sufficient to unblock a full live
  provider round-trip through the adapter). §5.3 requires either loosening
  `test_usage_limit_templates.py`'s assertions from exact equality to a form tolerant of an
  upstream-appended suffix, or accepting the wording as a documented pinned-template exception —
  a design decision this report does not make, since Requirement 8's contract for this unit is
  measurement only, not remediation (§1's "no code" boundary).

## 8. Environment integrity and the tooling carve-out (Requirement 8.2, 8.7, 13.2, 13.4)

- **The repository's environment and lock file are byte-identical to before this gate ran.**
  `git status --short` at the repository root, checked immediately before writing this report:
  ```
  ?? .worktrees/
  ```
  The only entry is the untracked worktree directory itself; `pyproject.toml`, `uv.lock`, and the
  repository's own `.venv` were never written to by anything in this report — every command that
  touched a dependency ran with an explicit `--project .worktrees/adapter-probe` (or a `cwd`
  pinned to that path), never against the repository root's own project context.
- **The gate is absent from pull-request CI.** `.github/workflows/pr.yml` and
  `.github/workflows/security.yml` contain no reference to `adapter`, `2.27`, or
  `pydantic-ai-slim` (confirmed by grep over both files, zero matches) — this gate has never been,
  and is not now, wired into any GitHub Actions workflow, per 8.7 (it needs a reachable Ollama
  instance and live LLM credits, neither of which PR CI has).
- **This is the one documented tooling carve-out.** Every other verification in this repository
  runs through `mise run <task>` against the project's own `.venv` (Principle 7). This gate is the
  single deliberate exception, and it runs *outside* `mise`/the project environment specifically
  *in service of* the 8.2 isolation criterion above — not against the "verification runs through
  `mise`" principle, since satisfying 8.2 at all requires an environment `mise run` cannot supply
  (a second `pyproject.toml`/`uv.lock`/`.venv` triple that isn't the project's own). This carve-out
  is recorded here, and in `plan.md`'s C9 component contract, as the only place it applies.
