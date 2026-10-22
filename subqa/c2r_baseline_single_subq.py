from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


C2R_SOURCE_MODEL_NAME = "Qwen2.5-VL-7B"
C2R_BASELINE_MODEL_NAME = "Baseline-SingleSubQ"
VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".webm"}


def build_c2r_source_path(c2r_root: str | Path, dataset_name: str, source_model_name: str = C2R_SOURCE_MODEL_NAME) -> Path:
    return Path(c2r_root) / "thesis" / dataset_name / source_model_name / "sub_qas.json"


def build_c2r_output_path(c2r_root: str | Path, dataset_name: str) -> Path:
    return Path(c2r_root) / dataset_name / C2R_BASELINE_MODEL_NAME / "sub_qas.json"


def load_c2r_manifest_records(source_path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(source_path).read_text())
    records = []
    for source_index, item in enumerate(payload):
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError("Each C2R manifest element must be a single-key dict.")
        ((question_id, record),) = item.items()
        records.append(
            {
                "source_index": int(source_index),
                "question_id": str(question_id),
                "main_question": record["main_question"],
                "media_path": record.get("media_path"),
            }
        )
    return records


def resolve_visual_source(media_path: Any, dummy_image_path: str) -> tuple[str, Any]:
    if media_path is None:
        return str(dummy_image_path), None
    if isinstance(media_path, list):
        if not media_path:
            raise ValueError("media_path list must not be empty.")
        return str(media_path[0]), media_path
    return str(media_path), media_path


def build_c2r_export_record(
    source_index: int,
    question_id: str,
    main_question: str,
    media_path: Any,
    sub_question: str,
) -> dict[str, Any]:
    return {
        "source_index": int(source_index),
        "question_id": str(question_id),
        "main_question": str(main_question),
        "media_path": media_path,
        "sub_q": [str(sub_question)],
        "sub_a": [],
    }


def merge_c2r_export_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_records = sorted(records, key=lambda record: int(record["source_index"]))
    merged = []
    seen_source_indices = set()
    for record in sorted_records:
        source_index = int(record["source_index"])
        if source_index in seen_source_indices:
            continue
        seen_source_indices.add(source_index)
        merged.append(
            {
                "question_id": record["question_id"],
                "main_question": record["main_question"],
                "media_path": record["media_path"],
                "sub_q": list(record["sub_q"]),
                "sub_a": list(record["sub_a"]),
            }
        )
    return merged
