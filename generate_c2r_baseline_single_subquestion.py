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
from subqa.baseline_single_subq import build_baseline_prompt, load_baseline_prompt_config
from subqa.c2r_baseline_single_subq import (
    C2R_SOURCE_MODEL_NAME,
    VIDEO_SUFFIXES,
    build_c2r_export_record,
    build_c2r_output_path,
    build_c2r_source_path,
    load_c2r_manifest_records,
    merge_c2r_export_records,
    resolve_visual_source,
)
from subqa.thesis_export import to_thesis_sub_qas

# imports for registry side effects
from lavis.datasets.builders import *  # noqa: F401,F403
from lavis.models import *  # noqa: F401,F403
from lavis.processors import *  # noqa: F401,F403
from lavis.runners import *  # noqa: F401,F403
from lavis.tasks import *  # noqa: F401,F403


DEFAULT_C2R_ROOT = "/home/ywjang/C2R"
DEFAULT_TMP_DIR = "/tmp/c2r_baseline_single_subq"


class C2RBaselineInferenceDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], vis_processor, text_processor, dummy_image_path: str):
        self.records = records
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.dummy_image_path = str(dummy_image_path)

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, visual_source: str) -> Image.Image:
        source_path = Path(visual_source)
        if source_path.suffix.lower() in VIDEO_SUFFIXES:
            import cv2

            capture = cv2.VideoCapture(str(source_path))
            ok, frame = capture.read()
            capture.release()
            if not ok:
                raise ValueError(f"Could not read first frame from video: {source_path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame)
        return Image.open(source_path).convert("RGB")

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        visual_source, output_media_path = resolve_visual_source(record["media_path"], self.dummy_image_path)
        image = self._load_image(visual_source)
        return {
            "image": self.vis_processor(image),
            "source_index": record["source_index"],
            "question_id": record["question_id"],
            "main_question": record["main_question"],
            "model_question": self.text_processor(record["main_question"]),
            "media_path": output_media_path,
        }

    @staticmethod
    def collater(samples: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "image": torch.stack([sample["image"] for sample in samples], dim=0),
            "source_index": [sample["source_index"] for sample in samples],
            "question_id": [sample["question_id"] for sample in samples],
            "main_question": [sample["main_question"] for sample in samples],
            "model_question": [sample["model_question"] for sample in samples],
            "media_path": [sample["media_path"] for sample in samples],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Baseline-SingleSubQ outputs for C2R datasets.")
    parser.add_argument("--cfg-path", default="instructBLIP_FT_vicuna7b.yaml")
    parser.add_argument("--questioner-checkpoint", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--baseline-prompt-path", default="baseline_prompts_single_subq.json")
    parser.add_argument("--c2r-root", default=DEFAULT_C2R_ROOT)
    parser.add_argument("--source-model-name", default=C2R_SOURCE_MODEL_NAME)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["MMLU", "MMMU", "MMLU-Pro", "StrategyQA", "EgoSchema"],
        choices=["MMLU", "MMMU", "MMLU-Pro", "StrategyQA", "EgoSchema"],
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--tmp-dir", default=DEFAULT_TMP_DIR)
    parser.add_argument("--dummy-image-path", default="/tmp/c2r_baseline_single_subq_dummy.png")
    return parser.parse_args()


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


def _ensure_dummy_image(path: str | Path, size: int = 224) -> str:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (size, size), color=(255, 255, 255)).save(path)
    return str(path)


def _load_records(args: argparse.Namespace, dataset_name: str) -> list[dict[str, Any]]:
    source_path = build_c2r_source_path(args.c2r_root, dataset_name, args.source_model_name)
    records = load_c2r_manifest_records(source_path)
    if args.limit is not None:
        records = records[: args.limit]
    return records


def run_dataset(
    args: argparse.Namespace,
    dataset_name: str,
    model,
    vis_processor,
    text_processor,
    prompt_config: dict[str, str],
    job_id: str,
    dummy_image_path: str,
) -> str | None:
    records = _load_records(args, dataset_name)
    rank = get_rank()
    world_size = get_world_size()
    sharded_records = records[rank::world_size]
    dataset = C2RBaselineInferenceDataset(
        records=sharded_records,
        vis_processor=vis_processor,
        text_processor=text_processor,
        dummy_image_path=dummy_image_path,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=C2RBaselineInferenceDataset.collater,
    )

    shard_records = []
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

        for source_index, question_id, main_question, media_path, sub_question in zip(
            batch["source_index"],
            batch["question_id"],
            batch["main_question"],
            batch["media_path"],
            outputs,
        ):
            shard_records.append(
                build_c2r_export_record(
                    source_index=source_index,
                    question_id=question_id,
                    main_question=main_question,
                    media_path=media_path,
                    sub_question=sub_question,
                )
            )

        if is_main_process() and (batch_index == 1 or batch_index % args.log_every == 0 or batch_index == total_batches):
            done = min(batch_index * args.batch_size * world_size, len(records))
            print(f"[{dataset_name}] processed {done}/{len(records)} examples")

    shard_dir = Path(args.tmp_dir) / job_id / dataset_name
    shard_path = shard_dir / f"rank{rank}.json"
    _save_json(shard_path, shard_records)

    if is_dist_avail_and_initialized():
        dist.barrier()

    if not is_main_process():
        return None

    merged_records = []
    for shard_rank in range(world_size):
        candidate = shard_dir / f"rank{shard_rank}.json"
        if candidate.exists():
            merged_records.extend(json.loads(candidate.read_text()))

    final_records = merge_c2r_export_records(merged_records)
    payload = to_thesis_sub_qas(final_records)
    final_path = build_c2r_output_path(args.c2r_root, dataset_name)
    _save_json(final_path, payload)
    print(f"Saved {dataset_name} results to {final_path}")
    return str(final_path)


def main() -> None:
    args = parse_args()
    cfg = _build_runtime_cfg(args.cfg_path)
    vis_processor, text_processor = _build_eval_processors(cfg)
    prompt_config = load_baseline_prompt_config(args.baseline_prompt_path)
    dummy_image_path = _ensure_dummy_image(args.dummy_image_path)
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
            dummy_image_path=dummy_image_path,
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
