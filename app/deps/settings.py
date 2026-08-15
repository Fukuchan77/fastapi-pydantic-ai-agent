"""Request-scoped settings dependency.

`create_app(settings=...)` injects an explicit `Settings` instance and the
lifespan publishes it as `app.state.settings`. A dependency that resolves
settings through process-global `get_settings()` instead reads whatever the
environment happens to say, so an application built with injected settings can
enforce one value while a request-path check applies another - the two halves of
a single decision coming from two different objects.

`get_request_settings` closes that gap for every `Depends(...)` site without
changing any dependant's signature: it is a drop-in replacement for
`Depends(get_settings)` that prefers `app.state.settings`.
"""

from fastapi import Request

from app.config import Settings
from app.config import get_settings


async def get_request_settings(request: Request) -> Settings:
    """Resolve the `Settings` the current request's application was built with.

    `app.state.settings` is authoritative. The `get_settings()` fallback covers
    only an application whose lifespan has not populated `app.state` (bare
    `FastAPI()` harnesses in unit tests); it can never override an injected
    value, because it is reached only when there is none.

    Args:
        request: The incoming request, whose `app.state` is consulted first.

    Returns:
        Settings: The application's settings.
    """
    settings = getattr(request.app.state, "settings", None)
    if isinstance(settings, Settings):
        return settings
    return get_settings()
