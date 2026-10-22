from __future__ import annotations

from typing import Any, Iterable, Sequence


THESIS_EXPORT_KEYS = ("question_id", "main_question", "media_path", "sub_q", "sub_a")


def build_answerer_generation_kwargs(
    use_nucleus_sampling: bool = False,
    num_beams: int = 5,
    max_length: int = 10,
    min_length: int = 1,
) -> dict[str, Any]:
    return {
        "use_nucleus_sampling": use_nucleus_sampling,
        "num_beams": int(num_beams),
        "max_length": int(max_length),
        "min_length": int(min_length),
    }


def build_answerer_rank_output_record(
    source_index: int,
    question_id: str,
    main_question: str,
    media_path: str,
    sub_questions: Sequence[str],
    sub_answers: Sequence[str],
) -> dict[str, Any]:
    if len(sub_questions) != len(sub_answers):
        raise ValueError(
            f"Mismatched sub-Q/sub-A counts for question_id={question_id}: "
            f"{len(sub_questions)} questions vs {len(sub_answers)} answers"
        )

    return {
        "source_index": int(source_index),
        "question_id": str(question_id),
        "main_question": str(main_question),
        "media_path": str(media_path),
        "sub_q": list(sub_questions),
        "sub_a": list(sub_answers),
    }


def merge_answerer_rank_output_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_records = sorted(records, key=lambda record: int(record["source_index"]))
    merged = []
    seen_source_indices = set()

    for record in sorted_records:
        source_index = int(record["source_index"])
        if source_index in seen_source_indices:
            continue
        seen_source_indices.add(source_index)
        merged.append({key: record[key] for key in THESIS_EXPORT_KEYS})

    return merged


def to_thesis_sub_qas(records: Sequence[dict[str, Any]]) -> list[dict[str, dict[str, Any]]]:
    payload = []
    for record in records:
        payload.append(
            {
                str(record["question_id"]): {
                    "main_question": record["main_question"],
                    "sub_q": list(record["sub_q"]),
                    "sub_a": list(record["sub_a"]),
                    "media_path": record["media_path"],
                }
            }
        )

    return payload
