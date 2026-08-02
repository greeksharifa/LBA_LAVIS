"""Model- and dataset-independent prompts for hierarchical Sub-QA."""

from collections.abc import Mapping, Sequence
from typing import Any, Optional, Tuple


SUPPORTED_MODALITIES = frozenset({"text", "image", "video"})


def _non_empty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return " ".join(value.split())


def _sample_context(sample: Mapping[str, Any]) -> Tuple[str, str]:
    if not isinstance(sample, Mapping):
        raise ValueError("sample must be a mapping")
    main_question = _non_empty_text(sample.get("main_q"), "sample.main_q")
    modality = _non_empty_text(sample.get("data_type"), "sample.data_type")
    if modality not in SUPPORTED_MODALITIES:
        raise ValueError(f"unsupported modality: {modality}")
    return main_question, modality


def _positive_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _normalized_texts(values: Sequence[str], label: str) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of strings")
    return tuple(
        _non_empty_text(value, f"{label}[{index}]")
        for index, value in enumerate(values)
    )


def _question_context(
    sample: Mapping[str, Any],
    *,
    ancestor_questions: Sequence[str] = (),
    parent_question: Optional[str] = None,
) -> list[str]:
    main_question, modality = _sample_context(sample)
    ancestors = _normalized_texts(ancestor_questions, "ancestor_questions")
    lines = [f"Main question: {main_question}", f"Modality: {modality}"]
    if ancestors:
        lines.append("Ancestor question path:")
        lines.extend(
            f"{index}. {question}"
            for index, question in enumerate(ancestors, start=1)
        )
    if parent_question is not None:
        lines.append(
            f"Parent question: {_non_empty_text(parent_question, 'parent_question')}"
        )
    return lines


def build_decomposition_prompt(
    sample: Mapping[str, Any],
    *,
    child_count: int,
    parent_question: Optional[str] = None,
    ancestor_questions: Sequence[str] = (),
) -> str:
    """Build a root or descendant child-question decomposition prompt."""
    count = _positive_count(child_count, "child_count")
    lines = [
        "Decompose the target into focused questions that help answer the main question.",
        *_question_context(
            sample,
            ancestor_questions=ancestor_questions,
            parent_question=parent_question,
        ),
        f"Generate exactly {count} distinct child questions.",
        "Output only a numbered list with one question per line and no other text.",
    ]
    return "\n".join(lines)


def build_question_repair_prompt(
    sample: Mapping[str, Any],
    *,
    child_count: int,
    existing_questions: Sequence[str],
    parent_question: Optional[str] = None,
    ancestor_questions: Sequence[str] = (),
) -> str:
    """Request only the questions missing from a partial decomposition."""
    count = _positive_count(child_count, "child_count")
    existing = _normalized_texts(existing_questions, "existing_questions")
    missing = count - len(existing)
    if missing <= 0:
        raise ValueError("question repair requires fewer existing questions than child_count")

    lines = [
        "Complete a partial question decomposition.",
        *_question_context(
            sample,
            ancestor_questions=ancestor_questions,
            parent_question=parent_question,
        ),
        "Existing valid questions:",
    ]
    lines.extend(
        f"{index}. {question}"
        for index, question in enumerate(existing, start=1)
    )
    lines.extend(
        (
            f"Generate exactly {missing} additional distinct child questions.",
            "Do not repeat or paraphrase an existing valid question.",
            "Target a different evidence dimension from the existing questions.",
            "A rewording that asks for the same answer is invalid.",
            "If the existing questions all ask for the final answer, ask for a "
            "prerequisite observation or concept instead.",
            "Output only a numbered list with one question per line and no other text.",
        )
    )
    return "\n".join(lines)


def build_direct_answer_prompt(
    sample: Mapping[str, Any],
    *,
    target_question: str,
    ancestor_questions: Sequence[str] = (),
) -> str:
    """Build a direct-answer prompt for any question node."""
    main_question, modality = _sample_context(sample)
    ancestors = _normalized_texts(ancestor_questions, "ancestor_questions")
    target = _non_empty_text(target_question, "target_question")
    lines = [
        "Answer the target question directly using the available modality context.",
        f"Main question: {main_question}",
        f"Modality: {modality}",
    ]
    if ancestors:
        lines.append("Ancestor question path:")
        lines.extend(
            f"{index}. {question}"
            for index, question in enumerate(ancestors, start=1)
        )
    lines.extend(
        (
            f"Target question: {target}",
            "Answer in a maximum of one sentence.",
        )
    )
    return "\n".join(lines)


def build_refined_answer_prompt(
    sample: Mapping[str, Any],
    *,
    target_question: str,
    child_qa: Sequence[Tuple[str, str]],
    direct_draft: Optional[str],
    condition_on_direct_suba: bool,
) -> str:
    """Build an internal-node answer prompt from selected immediate child QA."""
    main_question, modality = _sample_context(sample)
    target = _non_empty_text(target_question, "target_question")
    if not isinstance(condition_on_direct_suba, bool):
        raise ValueError("condition_on_direct_suba must be a boolean")
    if isinstance(child_qa, (str, bytes)) or not child_qa:
        raise ValueError("child_qa must be a non-empty sequence of question-answer pairs")

    normalized_child_qa = []
    for index, pair in enumerate(child_qa):
        if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)):
            raise ValueError(f"child_qa[{index}] must be a question-answer pair")
        if len(pair) != 2:
            raise ValueError(f"child_qa[{index}] must be a question-answer pair")
        normalized_child_qa.append(
            (
                _non_empty_text(pair[0], f"child_qa[{index}].question"),
                _non_empty_text(pair[1], f"child_qa[{index}].answer"),
            )
        )

    lines = [
        "Refine an answer to the target question using the selected child evidence.",
        f"Main question: {main_question}",
        f"Modality: {modality}",
        f"Target question: {target}",
    ]
    if condition_on_direct_suba:
        lines.extend(
            (
                "Direct draft:",
                _non_empty_text(direct_draft, "direct_draft"),
            )
        )
    lines.append("Selected child question-answer evidence:")
    for index, (question, answer) in enumerate(normalized_child_qa, start=1):
        lines.append(f"Child question {index}: {question}")
        lines.append(f"Child answer {index}: {answer}")
    lines.append("Answer in a maximum of one sentence.")
    return "\n".join(lines)
