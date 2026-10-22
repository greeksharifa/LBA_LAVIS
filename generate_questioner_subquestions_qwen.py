from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from subqa.qwen_questioner_subquestions import (
    VIDEO_SUFFIXES,
    build_qwen_user_messages,
    configure_qwen_processor,
    decode_qwen_generations,
)
from subqa.questioner_subquestions import (
    build_questioner_prompt,
    load_questioner_prompt_config,
)

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    def process_vision_info(messages, return_video_kwargs=True):
        has_media = any(item["type"] != "text" for item in messages[0]["content"])
        if has_media:
            raise ImportError("qwen_vl_utils is required for image/video inputs.")
        return None, None, {}


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_C2R_ROOT = "/home/ywjang/C2R"
DEFAULT_QIDS_ROOT = "/home/ywjang/LBA_LAVIS/qids"
DEFAULT_OUTPUT_ROOT = "/home/ywjang/Doctoral-Thesis/data"
MODEL_OUTPUT_NAME = "Baseline-Qwen2.5-VL-7B"
C2R_DATASET_NAMES = ["MMLU", "MMLU-Pro", "StrategyQA", "MMMU", "EgoSchema"]
C2R_DATASET_SPECS = {
    "MMLU": {
        "ann_candidates": [
            "{c2r_root}/MMLU/val",
            "{c2r_root}/MMLU",
            "/data/MMLU/MMLU/val",
        ],
    },
    "MMLU-Pro": {
        "ann_candidates": [
            "{c2r_root}/MMLU-Pro/test.csv",
            "{c2r_root}/MMLU-Pro/val.csv",
            "/data/MMLU/MMLU-Pro/test.csv",
        ],
    },
    "StrategyQA": {
        "ann_candidates": [
            "{c2r_root}/StrategyQA/C2R_test.json",
            "/data/StrategyQA/C2R_test.json",
        ],
    },
    "MMMU": {
        "ann_candidates": [
            "{c2r_root}/MMMU/mmmu_json/validation.json",
            "/data/MMMU/mmmu_json/validation.json",
        ],
    },
    "EgoSchema": {
        "ann_candidates": [
            "{c2r_root}/EgoSchema/C2R_test_400.csv",
            "/data/EgoSchema/C2R_test_400.csv",
        ],
        "vis_root_candidates": [
            "{c2r_root}/EgoSchema/good_clips_git",
            "/data/EgoSchema/good_clips_git",
        ],
    },
}


class QuestionerInferenceDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]

    @staticmethod
    def collater(samples: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "source_index": [sample["source_index"] for sample in samples],
            "question_id": [sample["question_id"] for sample in samples],
            "main_question": [sample["main_question"] for sample in samples],
            "media_path": [sample["media_path"] for sample in samples],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Questioner sub-questions with Qwen2.5-VL for C2R datasets.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--prompt-path", default="prompts.json")
    parser.add_argument("--c2r-root", default=DEFAULT_C2R_ROOT)
    parser.add_argument("--qids-root", default=DEFAULT_QIDS_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--dataset",
        choices=C2R_DATASET_NAMES,
        help="Run only one dataset. Equivalent to --datasets <name>.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=C2R_DATASET_NAMES,
        choices=C2R_DATASET_NAMES,
    )
    parser.add_argument("--num-sub-questions", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--min-pixels", type=int, default=28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    args = parser.parse_args()
    if args.dataset:
        args.datasets = [args.dataset]
    return args


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed_mode() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1 or dist.is_initialized():
        return

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", get_rank()))
        torch.cuda.set_device(local_rank)


def destroy_distributed_mode() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def resolve_torch_dtype(torch_dtype: str):
    if torch_dtype == "auto":
        return "auto"
    if not hasattr(torch, torch_dtype):
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
    return getattr(torch, torch_dtype)


def load_qwen_model_and_processor(
    model_name: str,
    torch_dtype: str,
    attn_implementation: str,
):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    dtype = resolve_torch_dtype(torch_dtype)
    model_kwargs = {
        "torch_dtype": dtype,
        "attn_implementation": attn_implementation,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": torch.cuda.current_device()}

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        **model_kwargs,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)
    return model, configure_qwen_processor(processor)


def _current_device_label() -> str:
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    return "cpu"


def _resolve_candidate_path(candidates: list[str], c2r_root: str) -> Path:
    resolved_candidates = [Path(candidate.format(c2r_root=c2r_root)).expanduser() for candidate in candidates]
    for candidate in resolved_candidates:
        if candidate.exists():
            return candidate
    return resolved_candidates[-1]


def _load_qids(dataset_name: str, qids_root: str) -> set[str]:
    qids_path = Path(qids_root) / f"{dataset_name}.json"
    payload = json.loads(qids_path.read_text())
    return {str(qid) for qid in payload["qids"]}


def _load_records_from_c2r_subqas(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    records = []
    for item in payload:
        if not isinstance(item, dict) or len(item) != 1:
            continue
        question_id, body = next(iter(item.items()))
        if not isinstance(body, dict):
            continue
        records.append(
            {
                "source_index": len(records),
                "question_id": str(question_id),
                "main_question": body.get("main_question"),
                "media_path": body.get("media_path"),
            }
        )
    return records


def _resolve_egoschema_video_path(vis_root: Path, video_name: str) -> str | None:
    for suffix in VIDEO_SUFFIXES:
        candidate = vis_root / f"{video_name}{suffix}"
        if candidate.exists():
            return str(candidate)
    return None


def _select_mmmu_media_paths(sample: dict[str, Any]) -> list[str]:
    question = sample["question"]
    image_paths = sample["image_path_list"]
    if "<image" not in question:
        return list(image_paths)

    selected = []
    for index, image_path in enumerate(image_paths, start=1):
        if f"<image {index}>" in question:
            selected.append(image_path)
    return selected or list(image_paths)


def _iter_mmlu_records(ann_dir: Path):
    for index, csv_path in enumerate(sorted(ann_dir.glob("*.csv"))):
        data = pd.read_csv(csv_path, names=["question", "A", "B", "C", "D", "answer"])
        for row_index, row in data.iterrows():
            yield {
                "qid": f"MMLU_{sum(len(pd.read_csv(p, names=['question', 'A', 'B', 'C', 'D', 'answer'])) for p in sorted(ann_dir.glob('*.csv'))[:index]) + row_index}",
                "main_question": row["question"],
                "media_path": None,
            }


def _load_mmlu_records(ann_dir: Path) -> list[dict[str, Any]]:
    records = []
    source_index = 0
    for csv_path in sorted(ann_dir.glob("*.csv")):
        data = pd.read_csv(csv_path, names=["question", "A", "B", "C", "D", "answer"])
        for _, row in data.iterrows():
            records.append(
                {
                    "source_index": source_index,
                    "question_id": f"MMLU_{source_index}",
                    "main_question": row["question"],
                    "media_path": None,
                }
            )
            source_index += 1
    return records


def load_c2r_records(
    dataset_name: str,
    c2r_root: str | Path,
    qids_root: str | Path,
    limit: int | None,
) -> list[dict[str, Any]]:
    spec = C2R_DATASET_SPECS[dataset_name]
    c2r_root = str(c2r_root)
    allowed_qids = _load_qids(dataset_name=dataset_name, qids_root=str(qids_root))

    if dataset_name == "MMLU":
        ann_dir = _resolve_candidate_path(spec["ann_candidates"], c2r_root)
        records = _load_mmlu_records(ann_dir)
        if not records:
            baseline_path = Path(c2r_root) / "MMLU" / "Baseline-SingleSubQ" / "sub_qas.json"
            if baseline_path.exists():
                records = _load_records_from_c2r_subqas(baseline_path)
    elif dataset_name == "MMLU-Pro":
        ann_path = _resolve_candidate_path(spec["ann_candidates"], c2r_root)
        data = pd.read_csv(ann_path)
        records = [
            {
                "source_index": index,
                "question_id": str(row["question_id"]),
                "main_question": row["question"],
                "media_path": None,
            }
            for index, (_, row) in enumerate(data.iterrows())
        ]
    elif dataset_name == "StrategyQA":
        ann_path = _resolve_candidate_path(spec["ann_candidates"], c2r_root)
        data = json.loads(ann_path.read_text())
        records = [
            {
                "source_index": index,
                "question_id": str(sample["qid"]),
                "main_question": sample["question"],
                "media_path": None,
            }
            for index, sample in enumerate(data)
        ]
    elif dataset_name == "MMMU":
        ann_path = _resolve_candidate_path(spec["ann_candidates"], c2r_root)
        data = json.loads(ann_path.read_text())
        records = [
            {
                "source_index": index,
                "question_id": str(sample["question_id"]),
                "main_question": sample["question"],
                "media_path": _select_mmmu_media_paths(sample),
            }
            for index, sample in enumerate(data)
        ]
    elif dataset_name == "EgoSchema":
        ann_path = _resolve_candidate_path(spec["ann_candidates"], c2r_root)
        vis_root = _resolve_candidate_path(spec["vis_root_candidates"], c2r_root)
        data = pd.read_csv(ann_path)
        qid_column = "qid" if "qid" in data.columns else "Unnamed: 0"
        records = [
            {
                "source_index": index,
                "question_id": str(row[qid_column]),
                "main_question": row["question"],
                "media_path": _resolve_egoschema_video_path(vis_root, row["video_name"]),
            }
            for index, (_, row) in enumerate(data.iterrows())
        ]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    filtered_records = []
    for record in records:
        if record["question_id"] not in allowed_qids:
            continue
        filtered_records.append(
            {
                "source_index": len(filtered_records),
                "question_id": record["question_id"],
                "main_question": record["main_question"],
                "media_path": record["media_path"],
            }
        )

    if limit is not None:
        filtered_records = filtered_records[:limit]

    return filtered_records


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _broadcast_job_id() -> str:
    payload = [None]
    if is_main_process():
        payload[0] = uuid.uuid4().hex[:12]
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _process_vision_inputs(message):
    try:
        result = process_vision_info(message, return_video_kwargs=True)
    except TypeError:
        result = process_vision_info(message)

    if isinstance(result, tuple):
        if len(result) == 3:
            image_input, video_input, video_kwargs = result
            return image_input, video_input, video_kwargs or {}
        if len(result) == 2:
            image_input, video_input = result
            return image_input, video_input, {}

    raise TypeError("Unsupported process_vision_info return value.")


def generate_sub_questions_for_batch(
    model,
    processor,
    model_questions: list[str],
    prompt_config: dict[str, str],
    num_sub_questions: int,
    num_beams: int,
    max_len: int,
    min_len: int,
    media_paths: list[Any] | None = None,
    image_paths: list[Any] | None = None,
) -> list[list[str]]:
    if media_paths is None:
        media_paths = image_paths
    if media_paths is None:
        raise ValueError("media_paths or image_paths must be provided.")
    generated = [[] for _ in model_questions]

    for _ in range(num_sub_questions):
        prompts = [
            build_questioner_prompt(question, previous, prompt_config)
            for question, previous in zip(model_questions, generated)
        ]
        messages = [
            build_qwen_user_messages(prompt=prompt, image_path=media_path)
            for prompt, media_path in zip(prompts, media_paths)
        ]
        rendered_prompts = [
            processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            for message in messages
        ]
        image_inputs = []
        video_inputs = []
        has_images = False
        has_videos = False
        processor_kwargs = {}
        for message in messages:
            image_input, video_input, video_kwargs = _process_vision_inputs(message)
            image_inputs.append(image_input)
            video_inputs.append(video_input)
            has_images = has_images or image_input is not None
            has_videos = has_videos or video_input is not None
            if video_kwargs:
                processor_kwargs.update(video_kwargs)

        input_kwargs = {
            "text": rendered_prompts,
            "padding": True,
            "return_tensors": "pt",
        }
        if has_images:
            input_kwargs["images"] = image_inputs
        if has_videos:
            input_kwargs["videos"] = video_inputs
        if processor_kwargs:
            input_kwargs.update(processor_kwargs)

        inputs = processor(**input_kwargs)
        if hasattr(inputs, "to"):
            inputs = inputs.to(model.device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_len,
            min_new_tokens=min_len,
            num_beams=num_beams,
        )
        outputs = decode_qwen_generations(
            processor=processor,
            generated_ids=generated_ids,
            attention_mask=inputs["attention_mask"],
        )
        for index, output in enumerate(outputs):
            generated[index].append(output)

    return generated


def run_dataset(
    args: argparse.Namespace,
    dataset_name: str,
    model,
    processor,
    prompt_config: dict[str, str],
    job_id: str,
) -> str | None:
    records = load_c2r_records(
        dataset_name=dataset_name,
        c2r_root=args.c2r_root,
        qids_root=args.qids_root,
        limit=args.limit,
    )
    output_dir = Path(args.output_root) / "C2R" / dataset_name / MODEL_OUTPUT_NAME
    rank = get_rank()
    world_size = get_world_size()
    sharded_records = records[rank::world_size]
    dataset = QuestionerInferenceDataset(sharded_records)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=QuestionerInferenceDataset.collater,
    )

    shard_records = []
    total_batches = len(dataloader)
    progress = tqdm(
        dataloader,
        total=total_batches,
        desc=f"{dataset_name} rank{rank}",
        disable=not is_main_process(),
    )
    for batch_index, batch in enumerate(progress, start=1):
        sub_questions = generate_sub_questions_for_batch(
            model=model,
            processor=processor,
            media_paths=batch["media_path"],
            model_questions=batch["main_question"],
            prompt_config=prompt_config,
            num_sub_questions=args.num_sub_questions,
            num_beams=args.num_beams,
            max_len=args.max_len,
            min_len=args.min_len,
        )

        for source_index, question_id, main_question, media_path, generated in zip(
            batch["source_index"],
            batch["question_id"],
            batch["main_question"],
            batch["media_path"],
            sub_questions,
        ):
            shard_records.append(
                {
                    "source_index": int(source_index),
                    "question_id": str(question_id),
                    "main_question": main_question,
                    "media_path": media_path,
                    "sub_q": list(generated),
                    "sub_a": [],
                }
            )

        if is_main_process() and (batch_index == 1 or batch_index % args.log_every == 0 or batch_index == total_batches):
            done = min(batch_index * args.batch_size * world_size, len(records))
            print(f"[{dataset_name}] processed {done}/{len(records)} examples")
        if hasattr(progress, "set_postfix_str"):
            progress.set_postfix_str(
                f"done={min(batch_index * args.batch_size, len(sharded_records))}/{len(sharded_records)}"
            )
    progress.close()

    shard_path = output_dir / f"{dataset_name}_questioner_subquestions_{job_id}_rank{rank}.json"
    _save_json(shard_path, shard_records)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if not is_main_process():
        return None

    merged_records = []
    for shard_rank in range(world_size):
        shard_file = output_dir / f"{dataset_name}_questioner_subquestions_{job_id}_rank{shard_rank}.json"
        if shard_file.exists():
            merged_records.extend(json.loads(shard_file.read_text()))

    final_payload = []
    seen_source_indices = set()
    for record in sorted(merged_records, key=lambda item: int(item["source_index"])):
        source_index = int(record["source_index"])
        if source_index in seen_source_indices:
            continue
        seen_source_indices.add(source_index)
        final_payload.append(
            {
                record["question_id"]: {
                    "main_question": record["main_question"],
                    "sub_q": list(record["sub_q"]),
                    "sub_a": list(record["sub_a"]),
                    "media_path": record["media_path"],
                }
            }
        )

    final_path = output_dir / "sub_qas.json"
    _save_json(final_path, final_payload)
    print(f"Saved {dataset_name} results to {final_path}")
    return str(final_path)


def main() -> None:
    args = parse_args()
    init_distributed_mode()
    prompt_config = load_questioner_prompt_config(args.prompt_path)
    model, processor = load_qwen_model_and_processor(
        model_name=args.model_name,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
    )
    job_id = _broadcast_job_id()
    final_paths = {}
    if is_main_process():
        print(
            f"Starting generation: rank={get_rank()} world_size={get_world_size()} device={_current_device_label()} datasets={args.datasets}"
        )

    try:
        for dataset_name in args.datasets:
            result = run_dataset(
                args=args,
                dataset_name=dataset_name,
                model=model,
                processor=processor,
                prompt_config=prompt_config,
                job_id=job_id,
            )
            if result is not None:
                final_paths[dataset_name] = result
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

        if is_main_process():
            print(json.dumps(final_paths, indent=2))
    finally:
        destroy_distributed_mode()


if __name__ == "__main__":
    main()
