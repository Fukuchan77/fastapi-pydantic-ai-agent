# Dependency runbook: closing the stale Dependabot backlog

Req 4.5, 4.6. This is a **human-run runbook**, not an automated procedure: closing a
GitHub pull request is a write operation against a shared system, and no task in
`003-pydantic-ai-v2-migration` executes `gh pr close` or any other GitHub write on the
implementer's behalf. `.github/dependabot.yml`'s `ignore`/`groups`/`open-pull-requests-limit`
entries are the only code deliverable; this document is the operator's checklist for
retiring the 14 PRs that predate them.

## Why these 14 exist and why none of them should be merged as-is

All 14 open Dependabot PRs are opened against the same stale base commit `956d9e2`, which
predates 61+ commits' worth of manual lock-file changes on `main` (current HEAD as of
writing: `5040586`). Comparing each PR's proposed target version against what `uv.lock`
already resolves today shows every PR falls into one of three buckets:

- **Superseded** (12 of 14): the current lock already resolves at or beyond the version
  the PR proposes. Merging would either no-op against the current lock or, because the
  PR's own lock diff was computed against the stale base, silently revert an already-newer
  resolution back down to the PR's older target.
- **Actively breaking** (#17): merging would raise the `starlette` upper bound past the
  pin that keeps the global rate limit alive.
- **Resolution-breaking downgrade** (#18): merging would pin `litellm` below the floor
  `pydantic-ai-litellm` itself requires.

## Per-PR disposition

| PR | Title | Base | Disposition | Rationale |
|---|---|---|---|---|
| #1 | bump `nltk` 3.9.3 → 3.9.4 | `956d9e2` | Close — superseded | `uv.lock` already resolves `nltk` 3.10.1. |
| #2 | bump `requests` 2.32.5 → 2.33.0 | `956d9e2` | Close — superseded | `uv.lock` already resolves `requests` 2.34.2. |
| #3 | bump `pygments` 2.19.2 → 2.20.0 | `956d9e2` | Close — superseded | `uv.lock` already resolves `pygments` 2.20.0; the PR target is already met. |
| #6 | bump `pillow` 12.1.1 → 12.2.0 | `956d9e2` | Close — superseded | `uv.lock` already resolves `pillow` 12.3.0. |
| #7 | bump `pytest` 9.0.2 → 9.0.3 (dev) | `956d9e2` | Close — superseded | `uv.lock` already resolves `pytest` 9.1.1. |
| #9 | bump `banks` 2.4.1 → 2.4.2 | `956d9e2` | Close — superseded | `uv.lock` already resolves `banks` 2.4.5. |
| #10 | bump `urllib3` 2.6.3 → 2.7.0 | `956d9e2` | Close — superseded | `uv.lock` already resolves `urllib3` 2.7.0; the PR target is already met. |
| #11 | bump `idna` 3.11 → 3.15 | `956d9e2` | Close — superseded | `uv.lock` already resolves `idna` 3.18. |
| #12 | bump `pydantic-ai-slim` 1.70.0 → 1.99.0 | `956d9e2` | Close — superseded | `uv.lock` already resolves `pydantic-ai-slim` 1.107.1, and the migration's own lock refresh (change unit 6) advances it further. The PR's base is far enough behind (1.70.0) that its diff is not a safe fast-forward. |
| #15 | bump `pyjwt` 2.12.1 → 2.13.0 | `956d9e2` | Close — superseded | `uv.lock` already resolves `pyjwt` 2.13.0; the PR target is already met. |
| #16 | bump `aiohttp` 3.13.3 → 3.14.1 | `956d9e2` | Close — superseded | `uv.lock` already resolves `aiohttp` 3.14.3. |
| #17 | bump `starlette` 0.52.1 → 1.3.1 | `956d9e2` | **Close — rejected, breaking** | `starlette<1.0` is a load-bearing pin (see `CLAUDE.md`/`AGENTS.md`/`pyproject.toml`). slowapi 0.1.10 is incompatible with starlette 1.x: exception-handler misdispatch, and `SlowAPIMiddleware` stops emitting `X-RateLimit-*`. Now structurally blocked from recurring by the `ignore: version-update:semver-major` entry for `starlette` in `.github/dependabot.yml`. |
| #18 | bump `litellm` 1.82.6 → 1.84.0 | `956d9e2` | **Close — rejected, breaking** | `pydantic-ai-litellm` declares its own floor of `litellm>=1.86.2`. `1.84.0` falls below that floor, so accepting this PR would pin a `litellm` version the adapter itself refuses to depend on — a downgrade into an unsupported combination, not merely a stale one. `litellm` is a transitive dependency (pulled in via `pydantic-ai-litellm`), so it is not a candidate for a direct `ignore` entry the way `starlette`/`fastapi` are; closing it is the only available control. |
| #19 | bump `pydantic-settings` 2.13.1 → 2.14.2 | `956d9e2` | Close — superseded | `uv.lock` already resolves `pydantic-settings` 2.14.2, matching the PR's own target; the migration's lock refresh (change unit 6) advances it further to 2.15.0. |

## Procedure (human-run)

For each PR above:

1. `gh pr view <number>` to reconfirm it is still open and still targets the stale base.
2. `gh pr close <number> --comment "<rationale from the table above>"`.
3. Do **not** re-run `uv lock --upgrade-package <name>` to "catch up" a superseded PR —
   the lock refresh in change unit 6 (constraint-preserving lock refresh) and the v2
   migration's own constraint edits (change unit 10) already carry the versions that
   matter; re-running per-PR risks re-introducing #17/#18's failure modes one dependency
   at a time.
4. After closing all 14, a fresh Dependabot run (next weekly `schedule.interval`) is
   expected to propose at most the grouped `minor-and-patch` PR plus any dependency this
   runbook did not anticipate. Any new proposal touching `starlette`, `fastapi`,
   `chromadb`, or `redis` majors, or `fastapi` minors, should never appear — if one does,
   `.github/dependabot.yml`'s `ignore` entries have regressed and
   `tests/unit/test_dependabot_config.py` should already be failing.

## Re-derivation note

The base commit, target-version comparisons, and open-PR list above were captured via
`gh pr list --state open --json number,title,baseRefOid` and manual inspection of
`uv.lock` at the time this runbook was written. If this document is consulted long after
that point, re-run those commands before trusting the per-PR "superseded" claims — the
Dependabot backlog is a moving target, and this table is a snapshot of one triage pass,
not a standing guarantee.
