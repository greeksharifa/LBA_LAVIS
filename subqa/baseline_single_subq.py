from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


BASELINE_PROMPT_KEY = "Baseline_SingleSubQ"
BASELINE_EXPORT_KEYS = ("question_id", "main_question", "media_path", "sub_q", "sub_a")


def load_baseline_prompt_config(
    prompt_path: str | os.PathLike[str] = "baseline_prompts_single_subq.json",
) -> dict[str, str]:
    prompt_file = Path(prompt_path)
    prompts = json.loads(prompt_file.read_text())
    return prompts[BASELINE_PROMPT_KEY]


def build_baseline_prompt(
    main_question: str,
    prompt_config: dict[str, str],
) -> str:
    return prompt_config["exemplar"] + prompt_config["suffix"].format(question=main_question)


def build_baseline_rank_output_record(
    record: dict[str, Any],
    sub_question: str,
) -> dict[str, Any]:
    return {
        "source_index": int(record["source_index"]),
        "question_id": str(record["question_id"]),
        "main_question": record["main_question"],
        "image_path": record["image_path"],
        "sub-questions": [str(sub_question)],
    }


def build_baseline_export_record(
    source_index: int,
    question_id: str,
    main_question: str,
    media_path: str,
    sub_question: str,
) -> dict[str, Any]:
    return {
        "source_index": int(source_index),
        "question_id": str(question_id),
        "main_question": str(main_question),
        "media_path": str(media_path),
        "sub_q": [str(sub_question)],
        "sub_a": [],
    }


def merge_baseline_export_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_records = sorted(records, key=lambda record: int(record["source_index"]))
    merged = []
    seen_source_indices = set()

    for record in sorted_records:
        source_index = int(record["source_index"])
        if source_index in seen_source_indices:
            continue
        seen_source_indices.add(source_index)
        merged.append({key: record[key] for key in BASELINE_EXPORT_KEYS})

    return merged
