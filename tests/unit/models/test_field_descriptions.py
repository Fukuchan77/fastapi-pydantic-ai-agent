"""Tests for Field descriptions in Pydantic models for OpenAPI documentation."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


def test_chat_request_field_descriptions_in_openapi(client):
    """Verify ChatRequest field descriptions appear in OpenAPI schema."""
    response = client.get("/openapi.json")
    assert response.status_code == 200

    schema = response.json()
    chat_request_schema = schema["components"]["schemas"]["ChatRequest"]

    # Check that message field has description
    assert "description" in chat_request_schema["properties"]["message"]
    message_desc = chat_request_schema["properties"]["message"]["description"]
    assert len(message_desc) > 0
    assert "user message" in message_desc.lower()

    # Check that session_id field has description
    assert "description" in chat_request_schema["properties"]["session_id"]
    session_desc = chat_request_schema["properties"]["session_id"]["description"]
    assert len(session_desc) > 0
    assert "session" in session_desc.lower()
    assert "stateless" in session_desc.lower() or "continuity" in session_desc.lower()


def test_chat_response_field_descriptions_in_openapi(client):
    """Verify ChatResponse field descriptions appear in OpenAPI schema."""
    response = client.get("/openapi.json")
    assert response.status_code == 200

    schema = response.json()
    chat_response_schema = schema["components"]["schemas"]["ChatResponse"]

    # Check that reply field has description
    assert "description" in chat_response_schema["properties"]["reply"]
    reply_desc = chat_response_schema["properties"]["reply"]["description"]
    assert len(reply_desc) > 0

    # Check that session_id field has description
    assert "description" in chat_response_schema["properties"]["session_id"]
    session_desc = chat_response_schema["properties"]["session_id"]["description"]
    assert len(session_desc) > 0

    # Check that tool_calls_made field has description
    assert "description" in chat_response_schema["properties"]["tool_calls_made"]
    tool_calls_desc = chat_response_schema["properties"]["tool_calls_made"]["description"]
    assert len(tool_calls_desc) > 0


def test_ingest_request_field_descriptions_in_openapi(client):
    """Verify IngestRequest field descriptions appear in OpenAPI schema."""
    response = client.get("/openapi.json")
    assert response.status_code == 200

    schema = response.json()
    ingest_request_schema = schema["components"]["schemas"]["IngestRequest"]

    # Check that chunks field has description
    assert "description" in ingest_request_schema["properties"]["chunks"]
    chunks_desc = ingest_request_schema["properties"]["chunks"]["description"]
    assert len(chunks_desc) > 0
    assert "chunk" in chunks_desc.lower() or "document" in chunks_desc.lower()


def test_ingest_response_field_descriptions_in_openapi(client):
    """Verify IngestResponse field descriptions appear in OpenAPI schema."""
    response = client.get("/openapi.json")
    assert response.status_code == 200

    schema = response.json()
    ingest_response_schema = schema["components"]["schemas"]["IngestResponse"]

    # Check that ingested field has description
    assert "description" in ingest_response_schema["properties"]["ingested"]
    ingested_desc = ingest_response_schema["properties"]["ingested"]["description"]
    assert len(ingested_desc) > 0


def test_rag_query_request_field_descriptions_in_openapi(client):
    """Verify RAGQueryRequest field descriptions appear in OpenAPI schema."""
    response = client.get("/openapi.json")
    assert response.status_code == 200

    schema = response.json()
    rag_query_request_schema = schema["components"]["schemas"]["RAGQueryRequest"]

    # Check that query field has description
    assert "description" in rag_query_request_schema["properties"]["query"]
    query_desc = rag_query_request_schema["properties"]["query"]["description"]
    assert len(query_desc) > 0

    # Check that max_retries field has description
    assert "description" in rag_query_request_schema["properties"]["max_retries"]
    max_retries_desc = rag_query_request_schema["properties"]["max_retries"]["description"]
    assert len(max_retries_desc) > 0
    assert "retr" in max_retries_desc.lower()


def test_rag_query_response_field_descriptions_in_openapi(client):
    """Verify RAGQueryResponse field descriptions appear in OpenAPI schema."""
    response = client.get("/openapi.json")
    assert response.status_code == 200

    schema = response.json()
    rag_query_response_schema = schema["components"]["schemas"]["RAGQueryResponse"]

    # Check that answer field has description
    assert "description" in rag_query_response_schema["properties"]["answer"]
    answer_desc = rag_query_response_schema["properties"]["answer"]["description"]
    assert len(answer_desc) > 0

    # Check that context_found field has description
    assert "description" in rag_query_response_schema["properties"]["context_found"]
    context_desc = rag_query_response_schema["properties"]["context_found"]["description"]
    assert len(context_desc) > 0

    # Check that search_count field has description
    assert "description" in rag_query_response_schema["properties"]["search_count"]
    search_count_desc = rag_query_response_schema["properties"]["search_count"]["description"]
    assert len(search_count_desc) > 0
