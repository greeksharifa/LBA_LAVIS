"""Parsing helpers for hierarchical Sub-QA generations."""

import re
import unicodedata
from typing import Any, List, Optional


_LIST_PREFIX = re.compile(
    r"^\s*(?:(?:\(?\d+\)?\s*[.)\]:-])|[-*\u2022])\s*"
)


def _positive_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("expected_count must be a positive integer")
    return value


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def _dedupe_key(question: str) -> str:
    key = question.casefold().rstrip()
    while key and unicodedata.category(key[-1]).startswith("P"):
        key = key[:-1].rstrip()
    return key


def parse_questions(text: str, *, expected_count: int) -> List[str]:
    """Parse, normalize, and first-order deduplicate generated questions."""
    count = _positive_count(expected_count)
    if not isinstance(text, str):
        raise ValueError("question generation text must be a string")

    questions = []
    seen = set()
    for raw_line in text.splitlines():
        question = _normalize_whitespace(_LIST_PREFIX.sub("", raw_line, count=1))
        if not question:
            continue
        key = _dedupe_key(question)
        if not key or key in seen:
            continue
        seen.add(key)
        questions.append(question)
        if len(questions) == count:
            break
    return questions


def parse_answer(text: str) -> Optional[str]:
    """Return a normalized non-empty answer, or ``None`` when invalid."""
    if not isinstance(text, str):
        raise ValueError("answer generation text must be a string")
    answer = _normalize_whitespace(text)
    return answer or None
