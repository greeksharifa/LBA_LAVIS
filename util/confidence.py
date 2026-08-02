"""Shared confidence normalization and deterministic candidate selection."""

import math


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number, got {value!r}")
    return value


def normalize_confidence(value, confidence_type):
    """Convert a supported raw metric to a higher-is-better score in [0, 1]."""
    value = _number(value, confidence_type)
    if confidence_type == "token_min_prob":
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"token_min_prob must be in [0, 1], got {value!r}"
            )
        return value
    if confidence_type == "seq_ppl":
        if value < 0.0:
            raise ValueError(f"seq_ppl must be non-negative, got {value!r}")
        return min(1.0, 1.0 / max(value, 1e-12))
    raise ValueError(f"unsupported confidence_type: {confidence_type!r}")


def select_first_maximum(values, confidences, confidence_type):
    """Return value, normalized confidence, and index for the first maximum."""
    if not isinstance(values, list) or not values:
        raise ValueError("candidate values must be a non-empty list")
    if not isinstance(confidences, list):
        raise ValueError("candidate confidence values must be a list")
    if len(values) != len(confidences):
        raise ValueError(
            "candidate value/confidence count mismatch: "
            f"values={len(values)}, confidences={len(confidences)}"
        )
    normalized = [
        normalize_confidence(value, confidence_type) for value in confidences
    ]
    index = max(range(len(normalized)), key=normalized.__getitem__)
    return values[index], normalized[index], index

