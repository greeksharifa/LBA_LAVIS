from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

from subqa.vqa_introspect_data import build_vqa_introspect_records, get_vqa_introspect_image_path


DATASET_OUTPUT_KEYS = ("question_id", "main_question", "image_path", "sub-questions")


def load_questioner_prompt_config(prompt_path: str | os.PathLike[str] = "prompts.json") -> dict[str, str]:
    prompt_file = Path(prompt_path)
    prompts = json.loads(prompt_file.read_text())
    return prompts["Questioner_MultipleSubQ"]


def build_questioner_prompt(
    main_question: str,
    previous_sub_questions: Sequence[str],
    prompt_config: dict[str, str],
) -> str:
    prompt = prompt_config["init_prompt"].format(main_question)

    if previous_sub_questions:
        prompt += prompt_config["after_prompt"]
        for idx, sub_question in enumerate(previous_sub_questions, start=1):
            if idx > 1:
                prompt += ", "
            prompt += prompt_config["pair_prompt"].format(idx, sub_question)
        prompt += "."

    return prompt


def normalize_vqa_introspect_records(
    json_data: dict[str, dict[str, Any]],
    vis_root: str,
    split: str = "val",
    sample_stride: int = 1,
) -> list[dict[str, Any]]:
    records = build_vqa_introspect_records(json_data, sample_stride=sample_stride)
    normalized = []

    for source_index, record in enumerate(records):
        normalized.append(
            {
                "source_index": source_index,
                "question_id": str(record["question_id"]),
                "main_question": record["question"],
                "image_path": os.path.abspath(
                    get_vqa_introspect_image_path(vis_root, int(record["image_id"]), split)
                ),
            }
        )

    return normalized


def normalize_aokvqa_records(
    annotations: Sequence[dict[str, Any]],
    vis_root: str,
) -> list[dict[str, Any]]:
    normalized = []

    for source_index, record in enumerate(annotations):
        normalized.append(
            {
                "source_index": source_index,
                "question_id": str(record["question_id"]),
                "main_question": record["question"],
                "image_path": os.path.abspath(os.path.join(vis_root, record["image"])),
            }
        )

    return normalized


def build_rank_output_record(
    record: dict[str, Any],
    sub_questions: Sequence[str],
) -> dict[str, Any]:
    return {
        "source_index": int(record["source_index"]),
        "question_id": str(record["question_id"]),
        "main_question": record["main_question"],
        "image_path": record["image_path"],
        "sub-questions": list(sub_questions),
    }


def merge_rank_output_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_records = sorted(records, key=lambda record: int(record["source_index"]))
    merged = []
    seen_source_indices = set()

    for record in sorted_records:
        source_index = int(record["source_index"])
        if source_index in seen_source_indices:
            continue
        seen_source_indices.add(source_index)
        merged.append({key: record[key] for key in DATASET_OUTPUT_KEYS})

    return merged
