# Context budget: staged rollout (X-7)

Ported from `vaz-ai-next/docs/context-budget.md`'s three-stage design, adapted to this
project's session-store architecture (see `docs/cross-repo-adoption-backlog.md` X-7).
This project's mechanical history trimmer (`app/stores/session_store/_trim.py`) already
existed before this document — it is Stage 0 below, not new work. What's new here is
naming the staged design explicitly and adding the Stage 1 seam a future compactor
plugs into.

## Stage 0 (current): mechanical trim only

Every `SessionStore.save_history()` call runs history through `trim_history()`
(`app/stores/session_store/_trim.py`), bounded by `session_max_messages` (default 1000).
No summarization, no token counting, no LLM involvement — just a pure, deterministic cut
that never orphans a tool-call pair and always keeps the system-prompt message
(`messages[0]`). See `CLAUDE.md`'s "Session history trimming" section for the full
invariant list.

On the RAG side, `PromptBuildingMixin._truncate_chunks`/`_truncate_hits`
(`app/workflows/rag_prompts.py`) bound retrieved context by character count before it
enters an LLM prompt, via `Settings.rag_prompt_max_chars` (default 15000 — the same
value every call site hardcoded before this setting existed, so Stage 0's behavior is
unchanged).

Neither mechanism inspects token counts or message *content* — both are structural
(message count, character count), which is why they're inexpensive and side-effect-free
enough to run on every save/prompt-build unconditionally.

## Stage 1 (opt-in, this PR): the compactor seam

`app/stores/session_store/_trim.py` now also defines:

```python
HistoryCompactor = Callable[[Sequence[ModelMessage]], Sequence[ModelMessage]]
```

Both `InMemorySessionStore` and `RedisSessionStore` accept an optional
`history_compactor: HistoryCompactor | None = None` constructor argument. When set,
`save_history()` applies it **before** `trim_history()` on every save — never after.
`trim_history()` always has final say over what is actually persisted, so a compactor
cannot itself violate the message-boundary/tool-call-pairing/non-empty-parts invariants;
it can only change what `trim_history()` sees as its input.

**The default (`None`) is byte-identical to Stage 0.** No call site constructs a store
with `history_compactor` set today, so nothing about current behavior changes from this
seam's mere existence — this mirrors the sibling `pydantic-ai-sandbox` deep-research
lane's `digest_fn` seam, whose Stage-0-equivalent is expressed as *"don't pass the
argument"*, not *"pass an identity function"* (`researcher.py`'s `digest_fn` keyword
defaults to the module's own pre-existing summarization function, so an un-injected call
is literally the old code path).

**What this PR does not ship**: an actual compactor implementation. The seam exists so
one can be added later (a window-pruning function, a call into the sibling repo's
`ResearchNote`-style structured note-taking, or a stand-alone summarizer) without
touching `_trim.py` or either `SessionStore` backend again. A future compactor:

- MUST NOT need to satisfy `trim_history()`'s invariants itself — those are re-enforced
  after it runs, regardless of what it returns.
- SHOULD prefer returning fewer, denser messages over returning malformed ones; a
  compactor that returns messages `trim_history()` then has to cut hard is wasted work,
  not a correctness problem.
- Is free to return its input unchanged (a Stage 0-equivalent no-op) — the type only
  constrains the shape, not that compaction actually happens on every call.

## Stage 2 (future, not started): automatic running summaries

The sibling document's Stage 2 — consolidating older conversation turns into a single
running-summary message, replacing them outright rather than merely dropping them —
is **not implemented**. Its own design gates the decision to build it on an
observation, not a schedule: only once `budget_exceeded` (or, here, the analogous
"trim actually had to cut something") becomes the *dominant* reason conversations hit a
limit, rather than an occasional edge case. Nothing in this codebase currently measures
that rate, so there is no evidence yet that Stage 2 is warranted. Revisit this stage if
`session_max_messages` trimming is observed to fire routinely on real traffic (today
there is no telemetry for that — adding it would itself be a prerequisite for deciding
whether to build Stage 2).

## Relationship to the RAG-side character budget

`Settings.rag_prompt_max_chars` is a separate, independent budget from
`session_max_messages` — one bounds a single RAG prompt's retrieved-context size, the
other bounds a session's total retained message count. Both are Stage 0 mechanisms
(character/message counting, no LLM involvement) and neither is in scope for a Stage 1
seam of its own in this PR: the truncation call sites already take a `max_chars`
parameter, so making that parameter configurable was a matter of wiring it to
`Settings` rather than needing a new plugin point.

## References

- `vaz-ai-next/docs/context-budget.md` — the Stage 0/1/2 staged-rollout design this
  document adapts (`WindowMessages` seam, `CHAT_TOKEN_BUDGET`/`MAX_STEPS` stop
  conditions, the `budget-exceeded`-dominance trigger for Stage 2).
- `pydantic-ai-sandbox/patterns/deep-research/src/patterns_deep_research/notes.py` —
  the `digest_fn` DI seam and `ResearchNote`/`distill_notes` structured note-taking this
  document's Stage 1 discussion draws its "don't pass the argument" default-seam pattern
  from. Not ported as a concrete compactor in this PR — see "What this PR does not
  ship" above.
- `app/stores/session_store/_trim.py` — Stage 0's mechanical trimmer, and the new
  `HistoryCompactor` type alias.
- `CLAUDE.md`'s "Session history trimming" section — the full invariant list
  `trim_history()` maintains regardless of what a Stage 1 compactor does first.
