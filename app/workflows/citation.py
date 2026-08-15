"""Citation ordering and validation for the Corrective RAG workflow.

Owns the rules for deterministic hit ordering (so citations are stable
across runs) and for validating that every cited id belongs to the set of
hit ids actually retrieved in the current workflow run.
"""

from app.models.rag import RetrievedHit
from app.workflows.exceptions import DanglingCitationError
from app.workflows.exceptions import EmptyCitationError


def order_hits(hits: list[RetrievedHit]) -> list[RetrievedHit]:
    """Order retrieved hits deterministically by (-score, chunk_id).

    Higher scores sort first; equal scores are broken by chunk_id ascending
    so ordering (and therefore citations) is stable across repeated runs.

    Args:
        hits: Retrieved hits to order.

    Returns:
        A new list of hits sorted by (-score, chunk_id). The input list is
        not mutated.
    """
    return sorted(hits, key=lambda hit: (-hit.score, hit.chunk_id))


def validate_citations(cited_ids: set[str], hit_ids: set[str]) -> None:
    """Validate that citations are non-empty and reference only known hit ids.

    Args:
        cited_ids: Ids cited by the generated answer.
        hit_ids: The full set of hit ids actually retrieved in the current run.

    Raises:
        EmptyCitationError: If ``cited_ids`` is empty.
        DanglingCitationError: If any id in ``cited_ids`` is absent from ``hit_ids``.
    """
    if not cited_ids:
        raise EmptyCitationError()

    unknown_ids = cited_ids - hit_ids
    if unknown_ids:
        raise DanglingCitationError(unknown_ids=unknown_ids, known_ids=hit_ids)
