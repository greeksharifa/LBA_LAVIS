from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataloader.inquirer_augmentation import (
    filter_generated_items,
    format_ratio_tag,
    removal_ratio,
)
from dataloader.inquirer_paths import (
    resolve_dataset_path,
    resolve_dataset_root,
    resolve_inquirer_path,
    resolve_source_root,
)


def build_dramaqa_split(
    *,
    source_root: str | Path | None = None,
    dataset_root: str | Path | None = None,
    filter_ratio: float = 0.5,
) -> Path:
    original_path = resolve_dataset_path(
        "dramaqa",
        "AnotherMissOhQA_train_set_ori_scsh.json",
        dataset_root=dataset_root,
    )
    scene_path = resolve_inquirer_path(
        "dramaqa",
        "AnotherMissOhQA_train_set_naive_sceneprob.json",
        source_root=source_root,
    )
    shot_path = resolve_inquirer_path(
        "dramaqa",
        "AnotherMissOhQA_train_set_naive_shotprob.json",
        source_root=source_root,
    )

    original_data = json.loads(original_path.read_text())
    scene_additional_data = json.loads(scene_path.read_text())
    shot_additional_data = json.loads(shot_path.read_text())
    data = (
        original_data
        + filter_generated_items(scene_additional_data, filter_ratio)
        + filter_generated_items(shot_additional_data, filter_ratio)
    )

    ratio_tag = format_ratio_tag(filter_ratio)
    output_path = resolve_dataset_path(
        "dramaqa",
        f"dramaqa_train_filtered_{ratio_tag}_Naive.json",
        dataset_root=dataset_root,
    )
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )
    return output_path


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the filtered DramaQA Flipped-VQA training split."
    )
    parser.add_argument(
        "--inquirer-source-root",
        default=str(resolve_source_root()),
    )
    parser.add_argument(
        "--dramaqa-root",
        default=str(resolve_dataset_root("dramaqa")),
    )
    parser.add_argument(
        "--filter-ratio",
        type=removal_ratio,
        default=0.5,
    )
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    build_dramaqa_split(
        source_root=args.inquirer_source_root,
        dataset_root=args.dramaqa_root,
        filter_ratio=args.filter_ratio,
    )
