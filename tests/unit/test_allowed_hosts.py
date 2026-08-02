"""Tests for Host header validation via TrustedHostMiddleware.

Starlette's router rebuilds redirect targets from the request's `Host` header and
`redirect_slashes` defaults to True, so before `TrustedHostMiddleware` was added
a request to any path with a trailing slash returned a 307 whose `Location`
pointed at a caller-supplied host. `/health/` needs no API key, so the primitive
was reachable unauthenticated.

These tests pin both halves of the fix: the middleware rejects untrusted hosts,
and `Settings` refuses a wildcard allow-list outside development (staging and
production both).
"""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings
from app.main import create_app
from tests.conftest import build_test_settings


def test_untrusted_host_is_rejected_before_redirect() -> None:
    """A spoofed Host must get 400 with no Location, not a 307 reflecting it."""
    app = create_app(settings=build_test_settings(allowed_hosts=["api.example.com"]))

    with TestClient(app) as client:
        response = client.get(
            "/health/",  # trailing slash triggers starlette's redirect_slashes path
            headers={"Host": "attacker.test"},
            follow_redirects=False,
        )

    assert response.status_code == 400, "Untrusted Host should be rejected outright"
    assert "location" not in response.headers, (
        "Rejected request must not emit a Location header; a redirect here is the "
        "open-redirect primitive this middleware exists to close"
    )
    assert "attacker.test" not in response.text, "Response must not echo the spoofed host"


def test_allowed_host_is_served() -> None:
    """A Host on the allow-list should be served normally."""
    app = create_app(settings=build_test_settings(allowed_hosts=["api.example.com"]))

    with TestClient(app) as client:
        response = client.get("/health", headers={"Host": "api.example.com"})

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_wildcard_subdomain_host_is_served() -> None:
    """Starlette's '*.example.com' pattern should match subdomains but not other domains."""
    app = create_app(settings=build_test_settings(allowed_hosts=["*.example.com"]))

    with TestClient(app) as client:
        allowed = client.get("/health", headers={"Host": "api.example.com"})
        rejected = client.get("/health", headers={"Host": "example.org"})

    assert allowed.status_code == 200, "Subdomain should match the *.example.com pattern"
    assert rejected.status_code == 400, "Unrelated domain should still be rejected"


def test_django_style_leading_dot_pattern_rejected() -> None:
    """A Django-style '.example.com' must fail fast, not silently match nothing.

    Starlette matches on `host == pattern or (pattern.startswith("*") and
    host.endswith(pattern[1:]))`, so a leading-dot pattern is treated as a
    literal hostname that no real Host header ever equals - which would lock out
    every request with a 400 that looks like a middleware bug.
    """
    with pytest.raises(ValidationError, match="would never match any host"):
        build_test_settings(allowed_hosts=[".example.com"])


def test_malformed_wildcard_pattern_rejected() -> None:
    """An interior wildcard is not supported by Starlette and must be rejected."""
    with pytest.raises(ValidationError, match="is not a valid wildcard"):
        build_test_settings(allowed_hosts=["api.*.example.com"])


def test_rejected_host_gets_no_www_redirect() -> None:
    """`www_redirect` is disabled, so a bare host must 400 rather than redirect.

    Starlette's www_redirect branch builds a `Location` from the request scope.
    Disabling it keeps every rejection a flat 400 with no Host-derived URL.
    """
    app = create_app(settings=build_test_settings(allowed_hosts=["www.example.com"]))

    with TestClient(app) as client:
        response = client.get(
            "/health",
            headers={"Host": "example.com"},
            follow_redirects=False,
        )

    assert response.status_code == 400, "Bare host should be rejected, not redirected to www"
    assert "location" not in response.headers, "No Location header should be emitted"


def test_bare_dot_wildcard_pattern_rejected() -> None:
    """`'*.'` (no label after the dot) must be rejected, not accepted as a quasi-wildcard.

    Starlette matches `pattern.startswith("*") and host.endswith(pattern[1:])`,
    so `'*.'` reduces to `host.endswith(".")` - which matches any FQDN written
    with a trailing dot (e.g. `example.com.`), not the empty-subdomain-only
    match the pattern's shape suggests.
    """
    with pytest.raises(ValidationError, match="is not a valid wildcard"):
        build_test_settings(allowed_hosts=["*."])


def test_wildcard_allowed_hosts_rejected_in_production() -> None:
    """`allowed_hosts=['*']` must fail fast when app_env is production."""
    with pytest.raises(ValidationError, match="allowed_hosts cannot contain '\\*'"):
        build_test_settings(app_env="production", allowed_hosts=["*"])


def test_wildcard_allowed_hosts_rejected_in_staging() -> None:
    """`allowed_hosts=['*']` must also fail fast in staging, not just production.

    Staging is public-reachable the same way production is, and
    `logging_config` already treats staging as production-like (INFO level,
    not DEBUG) - the Host-reflection primitive this setting closes doesn't
    care which non-development environment it's reached from.
    """
    with pytest.raises(ValidationError, match="allowed_hosts cannot contain '\\*'"):
        build_test_settings(app_env="staging", allowed_hosts=["*"])


def test_wildcard_mixed_with_a_real_host_still_rejected_in_production() -> None:
    """A wildcard alongside a named host must not "narrow" its way past the gate.

    An operator reading the field description ("Rejected as '*' outside
    development") might reasonably assume adding a real host alongside '*'
    makes the list more specific and therefore acceptable. It doesn't -
    `TrustedHostMiddleware` treats the presence of '*' anywhere in the list as
    allow-any (`self.allow_any = "*" in allowed_hosts`), so a mixed list is
    exactly as permissive as a bare `["*"]`.
    """
    with pytest.raises(ValidationError, match="allowed_hosts cannot contain '\\*'"):
        build_test_settings(app_env="production", allowed_hosts=["*", "api.example.com"])


def test_explicit_allowed_hosts_accepted_in_production() -> None:
    """Naming the served hosts should satisfy the production check."""
    settings = build_test_settings(app_env="production", allowed_hosts=["api.example.com"])

    assert settings.allowed_hosts == ["api.example.com"]


def test_explicit_allowed_hosts_accepted_in_staging() -> None:
    """Naming the served hosts should satisfy the staging check too."""
    settings = build_test_settings(app_env="staging", allowed_hosts=["api.example.com"])

    assert settings.allowed_hosts == ["api.example.com"]


def test_empty_allowed_hosts_rejected() -> None:
    """An empty allow-list would reject every request, so reject the config instead."""
    with pytest.raises(ValidationError, match="allowed_hosts must not be empty"):
        build_test_settings(allowed_hosts=[])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("api.example.com", ["api.example.com"]),
        ("api.example.com,*.internal.example.com", ["api.example.com", "*.internal.example.com"]),
        (
            " api.example.com , *.internal.example.com ",
            ["api.example.com", "*.internal.example.com"],
        ),
        (
            '["api.example.com","*.internal.example.com"]',
            ["api.example.com", "*.internal.example.com"],
        ),
    ],
)
def test_allowed_hosts_parses_string_forms(raw: str, expected: list[str]) -> None:
    """`ALLOWED_HOSTS` should accept the same shapes as `CORS_ORIGINS`."""
    settings = build_test_settings(allowed_hosts=raw)

    assert settings.allowed_hosts == expected


def test_allowed_hosts_rejects_malformed_json_array() -> None:
    """A value that looks like a JSON array but isn't should fail with a clear error."""
    with pytest.raises(ValidationError, match="looks like a JSON array but is not valid JSON"):
        build_test_settings(allowed_hosts='["api.example.com"')


def test_allowed_hosts_defaults_to_wildcard_outside_production() -> None:
    """The development default stays permissive so local runs and tests work unchanged."""
    settings: Settings = build_test_settings()

    assert settings.allowed_hosts == ["*"]


def test_allowed_hosts_list_input_drops_blank_entries() -> None:
    """A blank entry in a list input must not survive parsing.

    `["", "api.example.com"]` fed straight to `TrustedHostMiddleware` would add
    a pattern that equals an empty `Host` header, bypassing validation for any
    request that omits Host or sends it blank. The comma-separated string path
    already stripped and dropped empties; the list path must match.
    """
    settings = build_test_settings(allowed_hosts=["", "api.example.com", "  "])

    assert settings.allowed_hosts == ["api.example.com"]


def test_allowed_hosts_list_input_strips_whitespace() -> None:
    """A list entry with incidental whitespace must be trimmed, not rejected later.

    Untrimmed whitespace would fail the `TrustedHostMiddleware` match for every
    real request (`"api.example.com "  != "api.example.com"`), silently locking
    out the legitimate host rather than raising a config error.
    """
    settings = build_test_settings(allowed_hosts=[" api.example.com "])

    assert settings.allowed_hosts == ["api.example.com"]


def test_blank_list_entry_is_rejected_by_the_live_middleware() -> None:
    """End-to-end: a blank entry must not let an empty-Host request through.

    Regression guard for the gap the two tests above pin at the Settings layer -
    this exercises the actual `TrustedHostMiddleware` wired up by `create_app`.
    """
    app = create_app(settings=build_test_settings(allowed_hosts=["", "api.example.com"]))

    with TestClient(app) as client:
        response = client.get("/health", headers={"Host": ""})

    assert response.status_code == 400, (
        "Blank Host must be rejected, not matched by a stray '' pattern"
    )


def test_allowed_hosts_json_array_input_drops_blank_entries() -> None:
    """A blank entry inside a JSON array string must also be dropped."""
    settings = build_test_settings(allowed_hosts='["", "api.example.com"]')

    assert settings.allowed_hosts == ["api.example.com"]
