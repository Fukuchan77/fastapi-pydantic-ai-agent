"""Application configuration using Pydantic Settings.

Split into domain modules per `.sdd/steering/file-size-policy.md`
(`llm.py`, `store.py`, `security.py`, `observability.py`, `settings.py`);
this package re-exports the public surface so `from app.config import
Settings, get_settings` keeps working unchanged.
"""

from app.config.llm import _ALLOWED_LLM_PROVIDERS
from app.config.settings import Settings
from app.config.settings import get_settings


__all__ = ["_ALLOWED_LLM_PROVIDERS", "Settings", "get_settings"]
