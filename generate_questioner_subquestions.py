from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

import easydict
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import (
    get_rank,
    get_world_size,
    init_distributed_mode,
    is_dist_avail_and_initialized,
    is_main_process,
)
from lavis.common.logger import setup_logger
from lavis.processors import load_processor
from test_qar import _default_llm_model, _get_model, setup_seeds
from subqa.questioner_subquestions import (
    build_questioner_prompt,
    build_rank_output_record,
    load_questioner_prompt_config,
    merge_rank_output_records,
    normalize_aokvqa_records,
    normalize_vqa_introspect_records,
)

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


class QuestionerInferenceDataset(Dataset):
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
    parser = argparse.ArgumentParser(description="Generate Questioner sub-questions for validation datasets.")
    parser.add_argument("--cfg-path", default="instructBLIP_FT_vicuna7b.yaml")
    parser.add_argument("--questioner-checkpoint", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["vqa_introspect", "aok_vqa"],
        choices=["vqa_introspect", "aok_vqa"],
    )
    parser.add_argument("--prompt-path", default="prompts.json")
    parser.add_argument("--vqa-introspect-ann", default=DEFAULT_VQA_INTROSPECT_ANN)
    parser.add_argument("--aokvqa-ann", default=DEFAULT_AOKVQA_ANN)
    parser.add_argument("--coco-root", default=DEFAULT_COCO_ROOT)
    parser.add_argument("--vqa-introspect-output-dir", default=DEFAULT_VQA_INTROSPECT_OUTPUT_DIR)
    parser.add_argument("--aokvqa-output-dir", default=DEFAULT_AOKVQA_OUTPUT_DIR)
    parser.add_argument("--num-sub-questions", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def _first_existing_path(candidates: list[Path | str | None]) -> str | None:
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return str(candidate_path)
    return None


def _default_questioner_checkpoint() -> str:
    latest_local = sorted(
        Path("/home/ywjang/LBA_LAVIS/lavis/ywjang_output_questioner_multiple_subq").glob("*/checkpoint_best.pth")
    )
    latest_backup = sorted(
        Path("/home/ywjang/_backup_github/LBA_LAVIS/lavis/ywjang_output_questioner_multiple_subq").glob("*/checkpoint_best.pth")
    )
    checkpoint = _first_existing_path(
        [
            latest_local[-1] if latest_local else None,
            latest_backup[-1] if latest_backup else None,
            "/home/ywjang/_backup_github/LBA_LAVIS/lavis/ywjang_output/best_models/20230812121_Questioner_MultipleSubQ_epoch50/checkpoint_best.pth",
        ]
    )
    if checkpoint is None:
        raise FileNotFoundError("Could not find a Questioner checkpoint. Pass --questioner-checkpoint explicitly.")
    return checkpoint




def _looks_like_complete_llm_dir(path: Path) -> bool:
    return path.is_dir() and any(
        (path / name).exists()
        for name in (
            "pytorch_model.bin",
            "pytorch_model-00001-of-00002.bin",
            "model.safetensors",
            "model-00001-of-00002.safetensors",
        )
    )


def _default_llm_model_path() -> str:
    candidates = [
        Path("/models/lmsys/vicuna-7b-v1.1"),
        Path("/home/ywjang/models/lmsys/vicuna-7b-v1.1"),
        Path("/home/ywjang/models/eachadea/vicuna-7b-1.1"),
    ]
    for candidate in candidates:
        if _looks_like_complete_llm_dir(candidate):
            return str(candidate)
    return _default_llm_model()


def _build_runtime_cfg(cfg_path: str) -> Config:
    cfg = Config(easydict.EasyDict({"cfg_path": cfg_path, "options": []}))
    init_distributed_mode(cfg.run_cfg)
    setup_logger()
    setup_seeds(cfg)
    return cfg


def _build_eval_processors(cfg: Config):
    dataset_name = next(iter(cfg.datasets_cfg.keys()))
    dataset_cfg = cfg.datasets_cfg[dataset_name]
    vis_cfg = dataset_cfg.vis_processor.eval
    text_cfg = dataset_cfg.text_processor.eval
    vis_processor = load_processor(vis_cfg.name, vis_cfg)
    text_processor = load_processor(text_cfg.name, text_cfg)
    return vis_processor, text_processor


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


def _generate_sub_questions_for_batch(
    model,
    images: torch.Tensor,
    model_questions: list[str],
    prompt_config: dict[str, str],
    num_sub_questions: int,
    num_beams: int,
    max_len: int,
    min_len: int,
) -> list[list[str]]:
    generated = [[] for _ in model_questions]
    images = images.to(model.device, non_blocking=True)

    for _ in range(num_sub_questions):
        prompts = [
            build_questioner_prompt(question, previous, prompt_config)
            for question, previous in zip(model_questions, generated)
        ]
        outputs = model.generate(
            {"image": images, "prompt": prompts},
            use_nucleus_sampling=False,
            num_beams=num_beams,
            max_length=max_len,
            min_length=min_len,
        )
        for idx, output in enumerate(outputs):
            generated[idx].append(output)

    return generated


def run_dataset(
    args: argparse.Namespace,
    dataset_name: str,
    model,
    vis_processor,
    text_processor,
    prompt_config: dict[str, str],
    job_id: str,
) -> str | None:
    records, output_dir = _build_records(args, dataset_name)
    rank = get_rank()
    world_size = get_world_size()
    sharded_records = records[rank::world_size]
    dataset = QuestionerInferenceDataset(sharded_records, vis_processor, text_processor)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=QuestionerInferenceDataset.collater,
    )

    shard_records = []
    total_batches = len(dataloader)
    for batch_index, batch in enumerate(dataloader, start=1):
        sub_questions = _generate_sub_questions_for_batch(
            model=model,
            images=batch["image"],
            model_questions=batch["model_question"],
            prompt_config=prompt_config,
            num_sub_questions=args.num_sub_questions,
            num_beams=args.num_beams,
            max_len=args.max_len,
            min_len=args.min_len,
        )

        for source_index, question_id, main_question, image_path, generated in zip(
            batch["source_index"],
            batch["question_id"],
            batch["main_question"],
            batch["image_path"],
            sub_questions,
        ):
            shard_records.append(
                build_rank_output_record(
                    {
                        "source_index": source_index,
                        "question_id": question_id,
                        "main_question": main_question,
                        "image_path": image_path,
                    },
                    generated,
                )
            )

        if is_main_process() and (batch_index == 1 or batch_index % args.log_every == 0 or batch_index == total_batches):
            done = min(batch_index * args.batch_size * world_size, len(records))
            print(f"[{dataset_name}] processed {done}/{len(records)} examples")

    shard_path = output_dir / f"{dataset_name}_questioner_subquestions_{job_id}_rank{rank}.json"
    _save_json(shard_path, shard_records)

    if is_dist_avail_and_initialized():
        dist.barrier()

    if not is_main_process():
        return None

    merged_records = []
    for shard_rank in range(world_size):
        shard_file = output_dir / f"{dataset_name}_questioner_subquestions_{job_id}_rank{shard_rank}.json"
        if shard_file.exists():
            merged_records.extend(json.loads(shard_file.read_text()))

    final_records = merge_rank_output_records(merged_records)
    final_path = output_dir / f"{dataset_name}_questioner_subquestions_val_{job_id}.json"
    _save_json(final_path, final_records)
    print(f"Saved {dataset_name} results to {final_path}")
    return str(final_path)


def main() -> None:
    args = parse_args()
    cfg = _build_runtime_cfg(args.cfg_path)
    vis_processor, text_processor = _build_eval_processors(cfg)
    prompt_config = load_questioner_prompt_config(args.prompt_path)
    llm_model = args.llm_model or _default_llm_model_path()
    questioner_checkpoint = args.questioner_checkpoint or _default_questioner_checkpoint()
    finetuned = {"Questioner": questioner_checkpoint}
    questioner, _ = _get_model(
        cfg_path=args.cfg_path,
        finetuned=finetuned,
        role="Questioner",
        prompts=json.loads(Path(args.prompt_path).read_text()),
        llm_model=llm_model,
        build_runtime=False,
    )

    device = f"cuda:{cfg.run_cfg.gpu}" if torch.cuda.is_available() and getattr(cfg.run_cfg, 'distributed', False) else cfg.run_cfg.device
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
