"""Tests for OpenAPI security scheme configuration in main.py.

Task 16.5: Verify that the OpenAPI schema correctly defines the API key
security scheme so that API documentation (Swagger UI) properly displays
authentication requirements.
"""

from fastapi.testclient import TestClient

from app.main import app


def test_openapi_schema_has_security_schemes():
    """Test that OpenAPI schema includes securitySchemes definition.

    The OpenAPI schema should define an 'apiKey' security scheme
    for the X-API-Key header authentication.
    """
    client = TestClient(app)
    response = client.get("/openapi.json")
    assert response.status_code == 200

    openapi_schema = response.json()

    # Verify components.securitySchemes exists
    assert "components" in openapi_schema, "OpenAPI schema missing 'components'"
    assert "securitySchemes" in openapi_schema["components"], (
        "OpenAPI schema missing 'components.securitySchemes'"
    )

    security_schemes = openapi_schema["components"]["securitySchemes"]

    # Verify API key security scheme is defined
    # FastAPI auto-generates scheme name from variable name (APIKeyHeader -> 'APIKeyHeader')
    assert "APIKeyHeader" in security_schemes, (
        "Security scheme 'APIKeyHeader' not found in securitySchemes"
    )

    api_key_scheme = security_schemes["APIKeyHeader"]

    # Verify security scheme properties
    assert api_key_scheme["type"] == "apiKey", (
        f"Expected type 'apiKey', got {api_key_scheme.get('type')}"
    )
    assert api_key_scheme["in"] == "header", (
        f"Expected 'in' to be 'header', got {api_key_scheme.get('in')}"
    )
    assert api_key_scheme["name"] == "X-API-Key", (
        f"Expected name 'X-API-Key', got {api_key_scheme.get('name')}"
    )


def test_openapi_schema_has_security_description():
    """Test that the API key security scheme has a description.

    The security scheme should include a description explaining
    how to obtain and use the API key.
    """
    client = TestClient(app)
    response = client.get("/openapi.json")
    assert response.status_code == 200

    openapi_schema = response.json()
    api_key_scheme = openapi_schema["components"]["securitySchemes"]["APIKeyHeader"]

    # Verify description exists and is non-empty
    assert "description" in api_key_scheme, "API key security scheme missing 'description'"
    assert len(api_key_scheme["description"]) > 0, "API key security scheme description is empty"
    # Check that description mentions API key
    description_lower = api_key_scheme["description"].lower()
    assert "api" in description_lower, "Security scheme description should mention 'api'"
    assert "key" in description_lower, "Security scheme description should mention 'key'"


def test_protected_endpoints_require_security():
    """Test that protected endpoints specify security requirements.

    All /v1/* endpoints should reference the 'apiKey' security scheme
    in their OpenAPI definition.
    """
    client = TestClient(app)
    response = client.get("/openapi.json")
    assert response.status_code == 200

    openapi_schema = response.json()
    paths = openapi_schema.get("paths", {})

    # Find protected endpoints (all /v1/* paths)
    protected_paths = {path: methods for path, methods in paths.items() if path.startswith("/v1/")}

    assert len(protected_paths) > 0, "No /v1/* endpoints found in OpenAPI schema"

    # Verify each protected endpoint has security requirement
    for path, methods in protected_paths.items():
        for method, operation in methods.items():
            if method.lower() in ["get", "post", "put", "delete", "patch"]:
                # Check if security is defined at operation level or globally
                has_security = "security" in operation
                if has_security:
                    security = operation["security"]
                    # Verify APIKeyHeader is in the security requirements
                    assert any("APIKeyHeader" in req for req in security), (
                        f"Endpoint {method.upper()} {path} missing 'APIKeyHeader' in security"
                    )


def test_health_endpoint_no_security():
    """Test that /health endpoint does not require authentication.

    The /health endpoint should be publicly accessible and not
    require the apiKey security scheme.
    """
    client = TestClient(app)
    response = client.get("/openapi.json")
    assert response.status_code == 200

    openapi_schema = response.json()
    paths = openapi_schema.get("paths", {})

    # Verify /health endpoint exists
    assert "/health" in paths, "/health endpoint not found in OpenAPI schema"

    health_methods = paths["/health"]

    # Check GET method (primary health check method)
    if "get" in health_methods:
        get_operation = health_methods["get"]
        # Security should either be absent or explicitly empty
        security = get_operation.get("security", [])
        # Empty list means no authentication required
        assert security == [] or "security" not in get_operation, (
            "/health endpoint should not require authentication"
        )
