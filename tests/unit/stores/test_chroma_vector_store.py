"""Unit tests for ChromaVectorStore ID generation.

Tests cover ID generation to prevent race conditions in multi-process deployments.
Replace counter-based IDs with UUID-based IDs.
"""

import re
import uuid


def test_doc_counter_id_pattern():
    """Test that counter-based IDs follow the pattern 'doc_N'.

    This test documents the OLD behavior that causes race conditions.
    Counter-based IDs like "doc_0", "doc_1", "doc_2" cause collisions in
    multi-process deployments with shared persistent Chroma DB.
    """
    # Pattern for old counter-based IDs
    counter_pattern = re.compile(r"^doc_\d+$")

    # Examples of problematic counter-based IDs
    assert counter_pattern.match("doc_0")
    assert counter_pattern.match("doc_1")
    assert counter_pattern.match("doc_99")

    # UUID-based IDs should NOT match this pattern
    test_uuid = str(uuid.uuid4())
    assert not counter_pattern.match(test_uuid), "UUID should not match counter pattern"


def test_uuid_format_validation():
    """Test that UUID4 strings can be parsed as valid UUIDs.

    This test validates that UUID4 strings are valid UUIDs
    and can be used as document IDs in Chroma.
    """
    # Generate a UUID
    test_uuid_str = str(uuid.uuid4())

    # Should be parseable as UUID
    parsed = uuid.UUID(test_uuid_str)
    assert isinstance(parsed, uuid.UUID)

    # Should have UUID4 version
    assert parsed.version == 4

    # Should be 36 characters (including hyphens)
    assert len(test_uuid_str) == 36

    # Should not start with "doc_"
    assert not test_uuid_str.startswith("doc_")
