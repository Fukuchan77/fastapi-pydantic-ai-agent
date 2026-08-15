# Pydantic AI v2 behaviour notes: instrumentation attribute rename

Req 6.11. Documentation-only: no test or other document in this repository asserts on
usage-instrumentation attribute names today (verified — zero matches for
`InstrumentationSettings`, `use_aggregated_usage_attribute_names`, or `aggregated_usage`
outside this file), so this note is the only place recording the rename and the setting
that reverses it, for whoever next builds a Logfire dashboard or alert against this
codebase's traces.

## What changed

`pydantic_ai.models.instrumented.InstrumentationSettings` (installed version 2.30.0)
defaults `use_aggregated_usage_attribute_names` to `True`. When set, the **agent-run**
span's cumulative token-usage attributes are emitted under a custom
`gen_ai.aggregated_usage.*` namespace instead of the standard OpenTelemetry
`gen_ai.usage.*` names:

```python
def aggregated_usage_attributes(self, usage: UsageBase) -> dict[str, int]:
    attributes = usage.opentelemetry_attributes()
    if not self.use_aggregated_usage_attribute_names:
        return attributes
    return {key.replace("gen_ai.usage.", "gen_ai.aggregated_usage.", 1): value for key, value in attributes.items()}
```

The per-request `chat` spans nested under that run span keep `gen_ai.usage.*`
unchanged — only the parent run span's *cumulative* attributes are renamed. This is
paired with a second default change: the instrumentation data-format `version` moved
from `2` to `5` (`Literal[2, 3, 4, 5]`; version `1` no longer exists on 2.x). Versions
2–4 remain selectable as deprecated compatibility formats and emit a
`PydanticAIDeprecationWarning` when requested explicitly; this repository requests none
of them, so it now runs on format 5 implicitly.

## Why upstream made the change

From `InstrumentationSettings.__init__`'s own docstring: the flag "[d]efaults to `True`
to prevent double-counting in observability backends that aggregate span attributes
across parent and child spans." Summing `gen_ai.usage.*` across every span in a trace
would otherwise double-count each per-request `chat` span's usage against the parent
run span that shares the same attribute names. `gen_ai.aggregated_usage.*` is explicitly
a custom namespace, not part of the OpenTelemetry Semantic Conventions, and upstream
notes it may change again if OTel later standardizes one.

## Where this repository is affected

`app/observability.py::configure_logfire()` calls `logfire.instrument_pydantic_ai()`
with no arguments. `obj` defaults to `None`, which routes to
`Agent.instrument_all(settings)` with a `settings` built entirely from
`InstrumentationSettings` defaults (`logfire`'s own wrapper only overrides
`tracer_provider`/`meter_provider`/`logger_provider`; every other field, including
`use_aggregated_usage_attribute_names` and `version`, is left unset and falls through to
the class default). So this codebase inherited both default changes automatically the
moment task 7 bumped the `pydantic-ai-slim` constraint — no code in `app/` had to change
for the rename to take effect, which is exactly why it is invisible to `grep` and worth
recording here.

## Reconciling existing dashboards and queries

A dashboard, alert, or ad-hoc query built against `gen_ai.usage.*` on an **agent-run**
span (the per-request `chat` spans are unaffected) stops matching after this
repository's v2 bump. Two ways to reconcile, in order of preference:

1. **Point the query at `gen_ai.aggregated_usage.*` instead.** This is upstream's
   documented default going forward and needs no change in this repository.
2. **Pin the previous names**, if every downstream consumer cannot be updated at once,
   by passing the setting explicitly at the same call site:

   ```python
   logfire.instrument_pydantic_ai(use_aggregated_usage_attribute_names=False)
   ```

   in `app/observability.py::configure_logfire()`. This restores `gen_ai.usage.*` on the
   agent-run span, at the cost of reintroducing the double-counting risk the new default
   exists to avoid, in any backend that also sums the nested `chat` spans' `gen_ai.usage.*`
   into the same rollup.

This repository has made neither change as of this note: `configure_logfire()` remains
argument-free, so `gen_ai.aggregated_usage.*` under format version 5 is what any
dashboard built against this codebase's traces will observe.

## Reference

- `pydantic_ai.models.instrumented.InstrumentationSettings.__init__` (installed 2.30.0):
  `use_aggregated_usage_attribute_names: bool = True`, `version: Literal[2, 3, 4, 5] = 5`.
- `InstrumentationSettings.aggregated_usage_attributes()` performs the rename.
- `app/observability.py::configure_logfire()` is this repository's sole instrumentation
  call site (`logfire.instrument_pydantic_ai()`), and it passes neither setting
  explicitly.
