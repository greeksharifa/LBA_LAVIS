from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from generate_questioner_subquestions import (
    _build_eval_processors,
    _build_runtime_cfg,
    _default_llm_model_path,
    _default_questioner_checkpoint,
)
from lavis.common.dist_utils import (
    get_rank,
    get_world_size,
    is_dist_avail_and_initialized,
    is_main_process,
)
from test_qar import _get_model
from subqa.baseline_single_subq import (
    build_baseline_export_record,
    build_baseline_prompt,
    build_baseline_rank_output_record,
    load_baseline_prompt_config,
    merge_baseline_export_records,
)
from subqa.questioner_subquestions import (
    merge_rank_output_records,
    normalize_aokvqa_records,
    normalize_vqa_introspect_records,
)
from subqa.thesis_export import to_thesis_sub_qas

# imports for registry side effects
from lavis.datasets.builders import *  # noqa: F401,F403
from lavis.models import *  # noqa: F401,F403
from lavis.processors import *  # noqa: F401,F403
from lavis.runners import *  # noqa: F401,F403
from lavis.tasks import *  # noqa: F401,F403


DEFAULT_VQA_INTROSPECT_ANN = "/home/ywjang/data/VQA-Introspect/VQAIntrospect_valv1.0.json"
DEFAULT_AOKVQA_ANN = "/home/ywjang/data/aokvqa/annotations/aokvqa_v1p0_val.json"
DEFAULT_COCO_ROOT = "/home/ywjang/data/coco/images"
DEFAULT_VQA_INTROSPECT_OUTPUT_DIR = "/home/ywjang/data/VQA-Introspect/ywjang_sub_qas"
DEFAULT_AOKVQA_OUTPUT_DIR = "/home/ywjang/data/AOKVQA/ywjang_sub_qas"
DEFAULT_OUTPUT_ROOT = "/home/ywjang/LBA_LAVIS/thesis"
MODEL_FOLDER_NAME = "Baseline-SingleSubQ"
DATASET_FOLDER_NAMES = {
    "vqa_introspect": "VQA-Introspect",
    "aok_vqa": "AOK-VQA",
}


class BaselineInferenceDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], vis_processor, text_processor):
        self.records = records
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = Image.open(record["image_path"]).convert("RGB")
        return {
            "image": self.vis_processor(image),
            "source_index": record["source_index"],
            "question_id": record["question_id"],
            "main_question": record["main_question"],
            "model_question": self.text_processor(record["main_question"]),
            "image_path": record["image_path"],
        }

    @staticmethod
    def collater(samples: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "image": torch.stack([sample["image"] for sample in samples], dim=0),
            "source_index": [sample["source_index"] for sample in samples],
            "question_id": [sample["question_id"] for sample in samples],
            "main_question": [sample["main_question"] for sample in samples],
            "model_question": [sample["model_question"] for sample in samples],
            "image_path": [sample["image_path"] for sample in samples],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate baseline single sub-question outputs for validation datasets.")
    parser.add_argument("--cfg-path", default="instructBLIP_FT_vicuna7b.yaml")
    parser.add_argument("--questioner-checkpoint", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--baseline-prompt-path", default="baseline_prompts_single_subq.json")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["vqa_introspect", "aok_vqa"],
        choices=["vqa_introspect", "aok_vqa"],
    )
    parser.add_argument("--vqa-introspect-ann", default=DEFAULT_VQA_INTROSPECT_ANN)
    parser.add_argument("--aokvqa-ann", default=DEFAULT_AOKVQA_ANN)
    parser.add_argument("--coco-root", default=DEFAULT_COCO_ROOT)
    parser.add_argument("--vqa-introspect-output-dir", default=DEFAULT_VQA_INTROSPECT_OUTPUT_DIR)
    parser.add_argument("--aokvqa-output-dir", default=DEFAULT_AOKVQA_OUTPUT_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def _build_records(args: argparse.Namespace, dataset_name: str) -> tuple[list[dict[str, Any]], Path]:
    if dataset_name == "vqa_introspect":
        json_data = json.loads(Path(args.vqa_introspect_ann).read_text())
        records = normalize_vqa_introspect_records(
            json_data=json_data,
            vis_root=args.coco_root,
            split="val",
            sample_stride=args.sample_stride,
        )
        output_dir = Path(args.vqa_introspect_output_dir)
    elif dataset_name == "aok_vqa":
        annotations = json.loads(Path(args.aokvqa_ann).read_text())
        records = normalize_aokvqa_records(annotations=annotations, vis_root=args.coco_root)
        output_dir = Path(args.aokvqa_output_dir)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if args.limit is not None:
        records = records[: args.limit]

    return records, output_dir


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _broadcast_job_id() -> str:
    payload = [None]
    if is_main_process():
        payload[0] = uuid.uuid4().hex[:12]
    if is_dist_avail_and_initialized():
        dist.broadcast_object_list(payload, src=0)
    return payload[0]


def run_dataset(
    args: argparse.Namespace,
    dataset_name: str,
    model,
    vis_processor,
    text_processor,
    prompt_config: dict[str, str],
    job_id: str,
) -> dict[str, str] | None:
    records, output_dir = _build_records(args, dataset_name)
    rank = get_rank()
    world_size = get_world_size()
    sharded_records = records[rank::world_size]
    dataset = BaselineInferenceDataset(sharded_records, vis_processor, text_processor)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=BaselineInferenceDataset.collater,
    )

    raw_shard_records = []
    export_shard_records = []
    total_batches = len(dataloader)

    for batch_index, batch in enumerate(dataloader, start=1):
        images = batch["image"].to(model.device, non_blocking=True)
        prompts = [build_baseline_prompt(question, prompt_config) for question in batch["model_question"]]
        outputs = model.generate(
            {"image": images, "prompt": prompts},
            use_nucleus_sampling=False,
            num_beams=args.num_beams,
            max_length=args.max_len,
            min_length=args.min_len,
        )

        for source_index, question_id, main_question, image_path, sub_question in zip(
            batch["source_index"],
            batch["question_id"],
            batch["main_question"],
            batch["image_path"],
            outputs,
        ):
            raw_shard_records.append(
                build_baseline_rank_output_record(
                    {
                        "source_index": source_index,
                        "question_id": question_id,
                        "main_question": main_question,
                        "image_path": image_path,
                    },
                    sub_question,
                )
            )
            export_shard_records.append(
                build_baseline_export_record(
                    source_index=source_index,
                    question_id=question_id,
                    main_question=main_question,
                    media_path=image_path,
                    sub_question=sub_question,
                )
            )

        if is_main_process() and (batch_index == 1 or batch_index % args.log_every == 0 or batch_index == total_batches):
            done = min(batch_index * args.batch_size * world_size, len(records))
            print(f"[{dataset_name}] processed {done}/{len(records)} examples")

    raw_shard_path = output_dir / f"{dataset_name}_baseline_single_subquestion_{job_id}_rank{rank}.json"
    export_shard_path = output_dir / f"{dataset_name}_baseline_single_subquestion_thesis_{job_id}_rank{rank}.json"
    _save_json(raw_shard_path, raw_shard_records)
    _save_json(export_shard_path, export_shard_records)

    if is_dist_avail_and_initialized():
        dist.barrier()

    if not is_main_process():
        return None

    merged_raw_records = []
    merged_export_records = []
    for shard_rank in range(world_size):
        raw_path = output_dir / f"{dataset_name}_baseline_single_subquestion_{job_id}_rank{shard_rank}.json"
        export_path = output_dir / f"{dataset_name}_baseline_single_subquestion_thesis_{job_id}_rank{shard_rank}.json"
        if raw_path.exists():
            merged_raw_records.extend(json.loads(raw_path.read_text()))
        if export_path.exists():
            merged_export_records.extend(json.loads(export_path.read_text()))

    raw_final_records = merge_rank_output_records(merged_raw_records)
    raw_final_path = output_dir / f"{dataset_name}_baseline_single_subquestion_val_{job_id}.json"
    _save_json(raw_final_path, raw_final_records)

    thesis_records = merge_baseline_export_records(merged_export_records)
    thesis_payload = to_thesis_sub_qas(thesis_records)
    thesis_path = Path(args.output_root) / DATASET_FOLDER_NAMES[dataset_name] / MODEL_FOLDER_NAME / "sub_qas.json"
    _save_json(thesis_path, thesis_payload)

    print(f"Saved {dataset_name} raw results to {raw_final_path}")
    print(f"Saved {dataset_name} thesis results to {thesis_path}")
    return {
        "raw": str(raw_final_path),
        "thesis": str(thesis_path),
    }


def main() -> None:
    args = parse_args()
    cfg = _build_runtime_cfg(args.cfg_path)
    vis_processor, text_processor = _build_eval_processors(cfg)
    prompt_config = load_baseline_prompt_config(args.baseline_prompt_path)
    llm_model = args.llm_model or _default_llm_model_path()
    questioner_checkpoint = args.questioner_checkpoint or _default_questioner_checkpoint()
    finetuned = {"Questioner": questioner_checkpoint}
    questioner, _ = _get_model(
        cfg_path=args.cfg_path,
        finetuned=finetuned,
        role="Questioner",
        prompts={},
        llm_model=llm_model,
        build_runtime=False,
    )

    device = (
        f"cuda:{cfg.run_cfg.gpu}"
        if torch.cuda.is_available() and getattr(cfg.run_cfg, "distributed", False)
        else cfg.run_cfg.device
    )
    if torch.cuda.is_available():
        questioner = questioner.to(device)

    questioner.eval()
    job_id = _broadcast_job_id()
    final_paths = {}

    for dataset_name in args.datasets:
        result = run_dataset(
            args=args,
            dataset_name=dataset_name,
            model=questioner,
            vis_processor=vis_processor,
            text_processor=text_processor,
            prompt_config=prompt_config,
            job_id=job_id,
        )
        if result is not None:
            final_paths[dataset_name] = result
        if is_dist_avail_and_initialized():
            dist.barrier()

    if is_main_process():
        print(json.dumps(final_paths, indent=2))
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
