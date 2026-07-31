"""Runtime no-hardcoded-model-id guard (Req 14.2).

Mirrors the `no-hardcoded-model-id` pre-commit pygrep hook in
`.pre-commit-config.yaml` so CI still enforces the rule even when pre-commit
itself was skipped locally. Both places must stay in sync with
`app.config._ALLOWED_LLM_PROVIDERS`, the single source of truth for which
provider prefixes are recognized.

The rule only targets `app/` and `evals/` (production/runner code, where a
model id should always flow from `Settings`, never a literal). `tests/**` is
deliberately exempt: fixtures legitimately need a concrete "provider:model"
string to build a `Settings`/`TestModel` instance.

The pattern requires an assignment-like context (`=` or `:` directly before
the quoted string) so illustrative docstring mentions such as
`(e.g., "openai:gpt-4o")` - which appear in `app/config.py` and elsewhere -
are not false positives; only an actual literal assignment/keyword-argument/
dict-value is flagged.
"""

import re
from pathlib import Path

from app.config import _ALLOWED_LLM_PROVIDERS


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCANNED_DIRS = ("app", "evals")

_PATTERN = re.compile(r'[:=]\s*"(?:' + "|".join(_ALLOWED_LLM_PROVIDERS) + r'):[A-Za-z0-9_.-]+"')


def _iter_python_files() -> list[Path]:
    """Return every `.py` file under the scanned production directories."""
    files: list[Path] = []
    for directory in _SCANNED_DIRS:
        files.extend((_REPO_ROOT / directory).rglob("*.py"))
    return files


def _violations() -> list[str]:
    """Return `path:line` locations where a model id is hardcoded."""
    hits: list[str] = []
    for path in _iter_python_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if _PATTERN.search(line):
                hits.append(f"{path.relative_to(_REPO_ROOT)}:{lineno}")
    return hits


def test_pattern_actually_matches_a_hardcoded_model_id() -> None:
    """Guards against the regex silently matching nothing (a vacuously-passing guard)."""
    assert _PATTERN.search('llm_model = "openai:gpt-4o"')
    assert _PATTERN.search('model="anthropic:claude-3-5-sonnet-20241022"')
    assert not _PATTERN.search('# e.g., "openai:gpt-4o" is a valid identifier')


def test_no_hardcoded_model_id_in_app_or_evals() -> None:
    """No `app/**` or `evals/**` file may assign a literal "provider:model" string."""
    violations = _violations()
    assert not violations, f"Hardcoded model id(s) found (use Settings instead): {violations}"
