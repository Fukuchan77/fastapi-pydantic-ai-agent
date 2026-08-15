"""Shared placeholder detection for the package's secret-valued settings.

Every secret setting in `app/config/` rejects placeholder values so a deployment
that copied `.env.example` without filling a field in fails at startup rather
than running on a value published in this repository.

Enumerating per-field placeholder strings is not sufficient on its own. Each
validator carried its own hand-maintained set, and `session_signing_key` reused
`api_key`'s set verbatim without adding its own field-specific entry - so
`.env.example`'s `SESSION_SIGNING_KEY=your-session-signing-key-here` matched
nothing in that set and, at 30 characters, cleared the 16-character floor too.
It validated successfully, meaning a deployment that fixed only `API_KEY` (whose
own placeholder *is* enumerated, so it fails loudly) would silently run with a
publicly-known HMAC key and no session-ownership guarantee at all.

`SHAPE_PATTERN` closes that class of defect for every current and future secret
field: it matches the `...-here` convention `.env.example` uses throughout,
independently of whether anyone remembered to enumerate a given field's exact
string.
"""

import re
from collections.abc import Iterable


COMMON_PLACEHOLDERS: frozenset[str] = frozenset(
    {
        "your-api-key-here",
        "changeme",
        "change-me",
        "test-key",
        "test-token",
        "example",
        "replace-me",
        "insert-key-here",
        "api-key-here",
    }
)
"""Exact placeholder strings rejected for every secret field (case-insensitive)."""

SHAPE_PATTERN = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*[-_]here$", re.IGNORECASE)
"""Matches the `<words>-here` placeholder convention used across `.env.example`.

Covers `your-api-key-here`, `your-session-signing-key-here`, `your-token-here`,
`insert-key-here`, and any future field that follows the same convention. A real
secret ending in `-here` would be rejected too; that is the intended trade-off,
since the failure is a loud startup error an operator fixes in seconds, whereas
the miss it prevents is a silent, publicly-known key in production.
"""


def is_placeholder(value: str, extra: Iterable[str] = ()) -> bool:
    """Report whether `value` looks like an unfilled configuration placeholder.

    Args:
        value: The already-stripped secret value to classify.
        extra: Field-specific placeholder strings to reject in addition to
            `COMMON_PLACEHOLDERS` (e.g. `logfire_token`'s own `your-token-here`).

    Returns:
        True if `value` matches a known placeholder string (case-insensitive) or
        the `<words>-here` placeholder shape.
    """
    lowered = value.lower()
    if lowered in COMMON_PLACEHOLDERS or lowered in {item.lower() for item in extra}:
        return True
    return SHAPE_PATTERN.fullmatch(value) is not None
