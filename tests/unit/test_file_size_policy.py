"""Constitution Principle 6 file-size guard (`.sdd/steering/file-size-policy.md`).

Not tied to a spec requirement ID - enforces the steering doc's hard cap given
`app/config.py`'s cumulative growth across this spec's settings-adding tasks
(3, 6, 8, 10). Per plan.md's own Data Model description ("fails if any
`app/**`/`tests/**` file ... reaches the 1000-line hard cap"), only the
>=1000-line PROHIBITED tier is a hard failure; the 500-999 line "review"
tier is reported (not failed) here, since asserting a hard-coded
justification-comment convention that doesn't exist yet in this codebase
would fail on every pre-existing file already in that band
(`app/config.py`, `app/stores/vector_store.py`,
`app/workflows/corrective_rag.py`, ...) - none of which are in this task's
own boundary to annotate or split. Per the task's own instruction, a flagged
file without a justification note requires a follow-up task, not a doctored
test: see this task's Implementation Notes in tasks.md for the current
review-band file list and the corresponding follow-up.
"""

import warnings
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCANNED_DIRS = ("app", "tests")
_HARD_CAP = 1000
_REVIEW_BAND_START = 500


def _line_counts() -> dict[Path, int]:
    """Map every scanned `.py` file to its line count."""
    counts: dict[Path, int] = {}
    for directory in _SCANNED_DIRS:
        for path in (_REPO_ROOT / directory).rglob("*.py"):
            counts[path] = len(path.read_text(encoding="utf-8").splitlines())
    return counts


def test_scan_covers_at_least_one_file_in_each_directory() -> None:
    """Guards against the scan silently matching nothing."""
    counts = _line_counts()
    assert any(_REPO_ROOT / "app" in path.parents for path in counts)
    assert any(_REPO_ROOT / "tests" in path.parents for path in counts)


def test_no_python_file_reaches_the_1000_line_hard_cap() -> None:
    """No `app/**/*.py` or `tests/**/*.py` file may reach >= 1000 lines (PROHIBITED tier)."""
    violations = {
        path.relative_to(_REPO_ROOT): count
        for path, count in _line_counts().items()
        if count >= _HARD_CAP
    }
    assert not violations, (
        f"File(s) at or above the {_HARD_CAP}-line hard cap (must split before merging): "
        f"{violations}"
    )


def test_files_in_the_review_band_are_reported() -> None:
    """Files at 500-999 lines are flagged via a warning, not failed.

    A file appearing in pytest's warnings summary needs a justification note
    (or a split) before it grows into the 1000-line hard-cap band; see this
    test module's docstring for why that isn't enforced as a hard failure.
    """
    review_band = {
        path.relative_to(_REPO_ROOT): count
        for path, count in _line_counts().items()
        if _REVIEW_BAND_START <= count < _HARD_CAP
    }
    assert all(_REVIEW_BAND_START <= count < _HARD_CAP for count in review_band.values())
    if review_band:
        listing = ", ".join(
            f"{path} ({count} lines)" for path, count in sorted(review_band.items())
        )
        warnings.warn(
            f"File(s) in the file-size-policy review band (500-999 lines), "
            f"needing a justification note before reaching the 1000-line hard cap: {listing}",
            UserWarning,
            stacklevel=2,
        )
