"""Unit tests for app/workflows/citation.py.

Covers deterministic hit ordering (order_hits) and citation-id validation
(validate_citations) against the current run's retrieved hit set.
"""

import pytest

from app.models.rag import RetrievedHit
from app.workflows.citation import order_hits
from app.workflows.citation import validate_citations
from app.workflows.exceptions import DanglingCitationError
from app.workflows.exceptions import EmptyCitationError


def _hit(chunk_id: str, score: float, text: str = "text") -> RetrievedHit:
    return RetrievedHit(chunk_id=chunk_id, text=text, score=score)


class TestOrderHits:
    """Tests for order_hits()."""

    def test_orders_by_descending_score(self) -> None:
        """Hits should be sorted by score descending."""
        hits = [_hit("a::0000", 0.1), _hit("b::0000", 0.9), _hit("c::0000", 0.5)]

        ordered = order_hits(hits)

        assert [h.chunk_id for h in ordered] == ["b::0000", "c::0000", "a::0000"]

    def test_breaks_score_ties_by_chunk_id_ascending(self) -> None:
        """Equal-score hits should be ordered by chunk_id ascending for determinism."""
        hits = [_hit("z::0000", 0.5), _hit("a::0000", 0.5), _hit("m::0000", 0.5)]

        ordered = order_hits(hits)

        assert [h.chunk_id for h in ordered] == ["a::0000", "m::0000", "z::0000"]

    def test_empty_list_returns_empty_list(self) -> None:
        """Ordering an empty hit list should return an empty list."""
        assert order_hits([]) == []

    def test_does_not_mutate_input_list(self) -> None:
        """order_hits should return a new list, not sort in place."""
        hits = [_hit("b::0000", 0.1), _hit("a::0000", 0.9)]
        original_order = list(hits)

        order_hits(hits)

        assert hits == original_order

    def test_ordering_is_stable_across_repeated_calls(self) -> None:
        """Ordering the same hits repeatedly should be deterministic."""
        hits = [_hit("b::0000", 0.5), _hit("a::0000", 0.9), _hit("c::0000", 0.5)]

        first = [h.chunk_id for h in order_hits(hits)]
        second = [h.chunk_id for h in order_hits(hits)]

        assert first == second == ["a::0000", "b::0000", "c::0000"]


class TestValidateCitations:
    """Tests for validate_citations()."""

    def test_valid_subset_does_not_raise(self) -> None:
        """Citations that are a subset of the known hit ids should pass silently."""
        validate_citations(
            cited_ids={"memory::0000"},
            hit_ids={"memory::0000", "memory::0001"},
        )

    def test_all_known_ids_cited_does_not_raise(self) -> None:
        """Citing every known id should pass silently."""
        validate_citations(
            cited_ids={"memory::0000", "memory::0001"},
            hit_ids={"memory::0000", "memory::0001"},
        )

    def test_empty_citations_raises_empty_citation_error(self) -> None:
        """No citations at all should raise EmptyCitationError."""
        with pytest.raises(EmptyCitationError):
            validate_citations(cited_ids=set(), hit_ids={"memory::0000"})

    def test_unknown_citation_raises_dangling_citation_error(self) -> None:
        """A citation absent from the retrieved hit set should raise DanglingCitationError."""
        with pytest.raises(DanglingCitationError) as exc_info:
            validate_citations(
                cited_ids={"memory::0099"},
                hit_ids={"memory::0000", "memory::0001"},
            )

        assert exc_info.value.unknown_ids == frozenset({"memory::0099"})
        assert exc_info.value.known_ids == frozenset({"memory::0000", "memory::0001"})

    def test_mixed_valid_and_unknown_citations_lists_only_unknown(self) -> None:
        """DanglingCitationError should list only the unknown ids, not the valid ones."""
        with pytest.raises(DanglingCitationError) as exc_info:
            validate_citations(
                cited_ids={"memory::0000", "memory::0099"},
                hit_ids={"memory::0000", "memory::0001"},
            )

        assert exc_info.value.unknown_ids == frozenset({"memory::0099"})
