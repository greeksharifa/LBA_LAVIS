from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    scene_additional_data.sort(key=lambda item: item["perplex"], reverse=True)
    shot_additional_data.sort(key=lambda item: item["perplex"], reverse=True)

    scene_start = int(len(scene_additional_data) * filter_ratio)
    shot_start = int(len(shot_additional_data) * filter_ratio)
    data = (
        original_data
        + scene_additional_data[scene_start:]
        + shot_additional_data[shot_start:]
    )

    output_path = resolve_dataset_path(
        "dramaqa",
        "dramaqa_train_filtered05_Naive.json",
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
    parser.add_argument("--filter-ratio", type=float, default=0.5)
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    build_dramaqa_split(
        source_root=args.inquirer_source_root,
        dataset_root=args.dramaqa_root,
        filter_ratio=args.filter_ratio,
    )
