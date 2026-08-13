# Adapter-compatibility gate report — run 1 (`004-pydantic-ai-v2-unblock`, task 6)

Execution begun: 2026-08-13. This is the gate-evidence artifact Requirement 4.14 requires for
this execution of `004-pydantic-ai-v2-unblock`'s Requirement 4 gate — the re-run of the same
branch point `docs/adapter-probe-report.md` recorded as **FAILED** on 2026-08-11, now attempted
again after Requirements 1–3 and the run-end-strategy pin (tasks 2–5) landed on the 1.x lock.

**Status: tasks 6.1–6.4 complete.** Task 6.3 executed and graded the run; this write-up (task
6.4) restates that verdict in the fixed section order Requirement 4.14 and task 6.4's own text
require: complete resolved version set → interpreter version → lock-scope proof → per-stage
result for all six stages → the Ollama expected-live-test count in force at this execution,
stated explicitly → any documented stage-6 exception → failure classification (not applicable
here, since this execution is recorded PASSED). Task 6.2's ungraded pre-flight surfaced no
remediation, so the graded run in task 6.3 reused the same worktree §3 built rather than rebuilding
a fresh one, per this unit's own precondition ("only if 6.2 is clean"). Cross-reference:
`docs/adapter-probe-report.md` (the 2026-08-11 FAILED run this gate re-attempts; Requirement 4.15
forbids amending it — see the final section below for the pointer this write-up owes it). No
execution of `004`'s own Requirement 4 gate precedes this one: this is run 1.

**Verdict: PASSED.** Zero failures and zero errors across all six stages — a reversal of the
2026-08-11 report's 111-of-119 failures.

## 1. Complete resolved version set (Requirements 4.4, 4.14)

Captured via `uv run --project .worktrees/adapter-probe-004 python -m pip list --format=freeze`
inside the worktree's own venv (not `uv pip list`, which resolves against the ambient/default
project rather than an arbitrary `--project` target).

The two packages this gate exists to test, and their forced companion:

| Package | Repository's current lock | This gate's resolution |
|---|---|---|
| `pydantic-ai-slim` | 1.107.2 | **2.29.0** |
| `pydantic-graph` (forced companion) | 1.107.2 | **2.29.0** |
| `pydantic-ai-litellm` | 0.2.8 | 0.2.8 (unchanged — already its latest release satisfying the repository's bound) |
| `litellm` (adapter's own dependency) | 1.94.1 | 1.94.1 (unchanged) |

Every other package resolved identically to the repository's current `uv.lock` and current
`.venv` — confirmed two independent ways: the `git diff -- uv.lock` scope in §3, and a direct
`diff` between the repository venv's own freeze (`.venv/bin/python -m pip list --format=freeze`)
and this worktree's freeze, which differs on exactly two lines (`pydantic-ai-slim`,
`pydantic-graph`) out of 184 packages in each.

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
pydantic-ai-slim==2.29.0
pydantic_core==2.46.4
pydantic-graph==2.29.0
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

## 2. Interpreter version (Requirements 4.3, 4.14)

- **Repository's own environment**: `.venv/bin/python --version` → `Python 3.13.7`.
- **Gate's disposable environment**: pinned explicitly rather than left to the resolver's
  default — `uv sync --project .worktrees/adapter-probe-004 --python 3.13.7` — per the corrected
  methodology `docs/adapter-probe-report.md` §4 recorded (an unpinned sync there resolved CPython
  3.14.5 instead of the repository's own 3.13.7 and produced 41 confounding failures plus a
  multi-minute hang, none attributable to pydantic-ai).
- **Match confirmed**: both interpreters report **CPython 3.13.7**, verified via each
  interpreter's own `pip list --format=freeze` (§1 for the worktree; the repository's own `.venv`
  independently) rather than `uv pip list`, which resolves against the ambient/default project
  and would not distinguish the two.

## 3. Lock-scope proof (Requirements 4.1, 4.2, 4.4, 4.18)

- **Mechanism**: a separate `git worktree`, added from the repository root while on
  `003-pydantic-ai-v2-migration` at commit `1c938aa`:
  ```
  git worktree add ./.worktrees/adapter-probe-004 -b probe/adapter-pydantic-ai-2.27-004-run1
  ```
  The new branch (`probe/adapter-pydantic-ai-2.27-004-run1`) exists only to give the worktree a
  ref; it carries no commits of its own and none were made in it.
- **Dedicated virtual environment**: the worktree's own `.venv` at
  `.worktrees/adapter-probe-004/.venv`, created and used exclusively via `uv`'s `--project
  .worktrees/adapter-probe-004` flag (so every `uv`/`python` invocation ran against that
  environment, never the repository's own `.venv`).
- **Constraint edit, worktree-only (Requirement 4.1, 4.18)**: `pyproject.toml`'s
  `pydantic-ai-slim[logfire,openai]` bound was changed from `>=1.99.0,<2.0` to `>=2.27.0,<3.0`
  **inside the worktree only**, with a comment naming the task/run and recording that the
  repository's own bound stays on the 1.x line until task 7. `pydantic-ai-litellm`'s bound
  (`>=0.2.3,<1.0`) was left untouched — its current release (0.2.8) already satisfies it and
  resolves unchanged (§1).
- **Lock refresh, scope-preserving (Requirement 4.4)**: `uv lock --project
  .worktrees/adapter-probe-004 --upgrade-package pydantic-ai-slim --upgrade-package
  pydantic-ai-litellm` — deliberately *not* a bare `uv lock`, which would admit unrelated
  collateral version drift into the signal. `uv` reported:
  ```
  Resolved 204 packages in 743ms
  Updated pydantic-ai-slim v1.107.2 -> v2.29.0
  Updated pydantic-graph v1.107.2 -> v2.29.0
  ```
- **Byte-identical claim, proved not asserted (Requirement 4.4)**: `git diff -- uv.lock` from
  inside the worktree touches **only** the `pydantic-ai-slim` and `pydantic-graph` (its forced
  companion) version/hash fields, the `pyproject.toml` requires-dist line for `pydantic-ai-slim`,
  and the new `anyio` dependency edge those two packages now declare. No other package's entry
  changed. This is corroborated independently in §1 by diffing the two interpreters' own
  `pip list --format=freeze` output (184 packages each, exactly two differing lines).
- **No commit was made from inside the worktree at any point.** `git status --short` from inside
  the worktree, immediately after the lock refresh and venv re-sync:
  ```
   M pyproject.toml
   M uv.lock
  ```
- **The repository's own environment and lock file are unmodified.** `git status --short` from
  the repository root, checked immediately after building the worktree environment:
  ```
  ?? .worktrees/
  ```
  No tracked file changed. The repository's `.venv` was never invoked by any command in this
  gate — every `uv`/`python` call was explicit about `--project .worktrees/adapter-probe-004`.

## 4. Per-stage result — all six stages (Requirements 4.5–4.10, 4.17, 4.19)

Executed from the repository root against the §3 worktree's own environment, via
`uv run --project .worktrees/adapter-probe-004 ...` throughout — never `mise run`, so the
disposable environment stays outside the repository's own lock (§3). Live infrastructure
confirmed reachable before this run: a local Ollama instance with `granite4.1:8b` and
`llama3.2:latest` pulled (confirmed operationally by stages 5 and 6's successful live completions
below, since direct `curl` probing of the daemon was unavailable in this session's tool sandbox).
No file inside the worktree changed beyond the §3 baseline (`pyproject.toml`, `uv.lock`);
confirmed by `git status --short` immediately after this run. The repository root's own tracked
files are unmodified by this run (§3's confirmation still holds; this run only reads app/test
code and writes this evidence file). This gate is not, and has never been, wired into pull-request
CI (Requirement 4.17) — `.github/workflows/pr.yml` and `.github/workflows/security.yml` carry no
reference to `adapter`, `2.27`/`2.29`, or `pydantic-ai-slim`, since it needs a reachable Ollama
instance and live LLM credentials, neither of which PR CI has.

| Stage | Command | Result | Notes |
|---|---|---|---|
| 1. API lock test | `pytest tests/unit/test_pydantic_ai_api_lock.py -v` | **29 passed** | 0 failures, 0 errors |
| 2. Unit | `pytest tests/unit/ -q` | **1327 passed, 1 skipped** | Skip is `test_ollama_base_url_consistency.py::test_ollama_litellm_actual_request_url`, self-skipped under this hermetic tier's own constraint — identical to task 6.2's pre-flight, not a failure |
| 3. Integration | `pytest tests/integration/ -q` | **47 passed, 18 skipped** | Skips are the 6 Chroma + 7 Docker + 5 Redis opt-in/reachability-gated lanes, self-skipping identically to the 1.x resolution (Requirement 4.7) — identical counts to task 6.2's pre-flight |
| 4. E2E | `pytest tests/e2e/ -q` | **45 passed** | 0 failures, 0 errors; includes the rate-limiting enforcement canary; identical count to task 6.2's pre-flight |
| 5. Ollama-gated local lane | `EXPECT_LIVE_TESTS=5 pytest tests/local/ -v -m ollama` | **5 passed** | Collected exactly the pinned expected-live-test count against a **reachable** Ollama instance (see §5 for the count itself) — `test_llm_granite41.py` (3 cases) and `test_llm_llama32.py` (2 cases) all completed live inference successfully |
| 6. Evals | `python -m evals.runner` | **PASSED** | See detail below and §6 for the stage-6 exception check |

**Stage 6 detail:** `JUDGE_MODEL` was unset for this run, so the judge self-graded via the
agent-under-test's own model (`ollama:granite4.1:8b`); the `AUDIT:` warning task 4 requires was
correctly emitted, confirming self-grading is *survivable* here, not claimed to be *unbiased*.
All 3 golden cases (`capital-of-france`, `arithmetic-addition`, `largest-planet`) received numeric
ratings on both axes (no `"Unknown"`): `outcome aggregate: 5.0`, `behavior aggregate: 5.0`, both
`≥ 3.0`. The run terminated normally with at least one numeric rating on each axis, satisfying
Requirement 4.19.1 and closing the graded-nothing false-green case it exists to prevent.

Operational note: `evals/runner.py`'s `_log_report()` uses `logger.info()`, and neither
`evals/runner.py` nor this ad-hoc invocation calls `configure_logging()` (that is wired only into
`app.lifespan`'s startup path), so the per-case ratings and aggregates are invisible at the
default root-logger level (`WARNING`). Only the `AUDIT:` line (`logger.warning`) surfaces without
extra configuration; the exit code (`0`/`1` from `report.passed`) is still correct either way, but
confirming Requirement 4.19.1's "at least one numeric rating per axis" clause required re-running
with `logging.basicConfig(level=logging.INFO)` to surface the aggregates. This is a pre-existing
property of the pre-push-only tooling on both the 1.x and 2.x lines, not a v2-specific finding.

**Environment-configuration detour (not part of the gate signal):** the worktree's `Settings()`
construction initially failed twice before this run could start — `session_signing_key` missing,
then (after that was set) `judge_model=""` failing its `'provider:model'` format check. Both
failures were confirmed to reproduce **identically against the repository's own 1.x `.venv`**
(same command, same error), proving they are pre-existing local `.env` configuration gaps
unrelated to the pydantic-ai v2 bump. The operator (not this implementer) corrected both directly
in `.env`. Recorded here only because it shows the corrective steps a reader would otherwise
wonder about; it is not a gate finding and does not affect this run's verdict.

## 5. Ollama expected-live-test count in force at this execution (Requirements 4.8, 4.9, 4.14)

**5** — matching the literal `.githooks/pre-push` pin (`EXPECT_LIVE_TESTS=5 mise run test:local`)
and `.github/workflows/pr.yml`'s `EXPECT_LIVE_TESTS: "5"` for the Redis lane, unchanged as of this
execution. Stage 5 (§4) collected and passed exactly this pinned count against a reachable Ollama
instance, so this stage was not recorded green by collecting nothing (Requirement 4.8). No
stage-5 drift-guard constant exists on this lane yet (unlike `tests/support/redis.py`'s
`REDIS_LIVE_TEST_COUNT` or `tests/support/chroma.py`'s `CHROMA_LIVE_TEST_COUNT`) — that is task
8.3/8.4's still-pending deliverable, later in this spec's own sequencing (Requirement 9.1 places
it strictly after this gate's PASS).

This statement exists because task 8 raises this count afterwards, and per Requirement 4.9 the
gate's meaning must not drift silently once it does: any reader of this artifact, at any later
date, can confirm this execution's stage 5 was graded against **5**, not against whatever count is
in force when they read it.

## 6. Documented stage-6 exception (Requirement 4.19.2)

**None.** Requirement 4.19.2 requires a documented exception only where a run completes but an
axis aggregate falls below the harness's 3.0 threshold. Both axes cleared that threshold outright
in this execution — `outcome aggregate: 5.0`, `behavior aggregate: 5.0` (§4) — so no exception is
invoked, and none is recorded here. This section is retained in the fixed order regardless of
outcome, so a future run that *does* need one has a stable place to record it.

## 7. Failure classification (Requirement 4.14)

**Not applicable to this execution.** The verdict is PASSED (§4): zero failures and zero errors
across all six stages, so there is no failing assertion to classify to a root cause. This section
is retained in the fixed order regardless of outcome, so a future FAILED execution's classification
lands in the same place a reader would already know to look.

## Cross-references

- **Original report**: `docs/adapter-probe-report.md`, executed 2026-08-11, recorded **FAILED**
  under `003-pydantic-ai-v2-migration`'s Requirement 8 — three failure signatures (the
  `ModelProfile` shape change, the native-output capability read, and the `UsageLimitExceeded`
  message-template wording). Requirements 1–3 and the run-end-strategy pin (tasks 2–5 of this
  spec) each close one of those three before this re-run, which is why this execution's stages 1–4
  (§4) reproduce zero of the original 111 blocked-construction failures and zero of the original
  7 template-wording failures. Requirement 4.15 forbids amending that report; task 6.5 added the
  dated forward-pointer line there and repointed the tracked citation slots in
  `CLAUDE.md`/`AGENTS.md` at this run.
- **Preceding runs of this spec's own Requirement 4 gate**: none. This is run 1 — the first
  execution of `004-pydantic-ai-v2-unblock`'s own gate.

## Next steps

- Task 6.5 is done: `docs/adapter-probe-report.md` carries the dated forward-pointer line, and
  `CLAUDE.md`/`AGENTS.md` cite this run (Requirement 4.16).
- Given the PASSED verdict, tasks 7 and 8 are now unblocked (Requirement 9.1): task 7 may begin the
  constraint bump, v2 migration, and Redis key-prefix cutover.
