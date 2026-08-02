"""Deterministic evidence subset and confidence selection helpers."""

from itertools import combinations, islice
from typing import Mapping, Sequence

from util.confidence import select_first_maximum


def _positive_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def evidence_subsets(children: Sequence, m: int, k: int) -> tuple:
    """Return the first ``k`` lexicographic size-``m`` evidence combinations."""
    if isinstance(children, (str, bytes)) or not isinstance(children, Sequence):
        raise ValueError("children must be an ordered sequence")
    m = _positive_integer(m, "m")
    k = _positive_integer(k, "k")
    return tuple(islice(combinations(tuple(children), m), k))


def select_best_candidate(candidates: Sequence[Mapping], confidence_type: str):
    """Select the first normalized maximum among valid candidates.

    Returns ``(candidate, normalized_score, original_index)`` or ``None`` when
    no candidate has status ``valid``.
    """
    if isinstance(candidates, (str, bytes)) or not isinstance(
        candidates, Sequence
    ):
        raise ValueError("candidates must be an ordered sequence")

    valid_candidates = []
    raw_confidences = []
    original_indices = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"candidate at index {index} must be a mapping")
        status = candidate.get("status")
        if status == "invalid":
            continue
        if status != "valid":
            raise ValueError(
                f"candidate at index {index} has unsupported status {status!r}"
            )
        confidence = candidate.get("confidence")
        if not isinstance(confidence, Mapping) or confidence_type not in confidence:
            raise ValueError(
                f"valid candidate at index {index} is missing "
                f"confidence metric {confidence_type!r}"
            )
        valid_candidates.append(candidate)
        raw_confidences.append(confidence[confidence_type])
        original_indices.append(index)

    if not valid_candidates:
        return None
    selected, normalized_score, valid_index = select_first_maximum(
        valid_candidates, raw_confidences, confidence_type
    )
    return selected, normalized_score, original_indices[valid_index]
