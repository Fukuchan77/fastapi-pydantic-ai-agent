"""Unit tests for app.deps module initialization and exports."""


def test_deps_module_exports_verify_api_key() -> None:
    """Test that verify_api_key is exported from app.deps package.

    Task 20.9: Ensures the deps package properly exports its public API.
    """
    from app.deps import verify_api_key

    # Should be callable
    assert callable(verify_api_key)


def test_deps_module_exports_get_rag_workflow() -> None:
    """Test that get_rag_workflow is exported from app.deps package.

    Task 20.9: Ensures the deps package properly exports its public API.
    """
    from app.deps import get_rag_workflow

    # Should be callable
    assert callable(get_rag_workflow)


def test_deps_module_has_all_attribute() -> None:
    """Test that __all__ is defined for explicit public API declaration.

    Task 20.9: Ensures the deps package declares its public interface.
    """
    import app.deps

    assert hasattr(app.deps, "__all__")
    assert isinstance(app.deps.__all__, list)
    assert "verify_api_key" in app.deps.__all__
    assert "get_rag_workflow" in app.deps.__all__


def test_deps_module_does_not_export_private_members() -> None:
    """Test that private/implementation details are not in __all__.

    Task 20.9: Ensures only public API is exported.
    """
    import app.deps

    # Should not export logger, APIKeyHeader, etc.
    assert "logger" not in app.deps.__all__
    assert "api_key_header" not in app.deps.__all__
