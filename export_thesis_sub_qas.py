from __future__ import annotations

import argparse
import json
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
)
from lavis.common.dist_utils import (
    get_rank,
    get_world_size,
    is_dist_avail_and_initialized,
    is_main_process,
)
from test_qar import _get_model
from subqa.thesis_export import (
    build_answerer_rank_output_record,
    merge_answerer_rank_output_records,
    to_thesis_sub_qas,
)

# imports for registry side effects
from lavis.datasets.builders import *  # noqa: F401,F403
from lavis.models import *  # noqa: F401,F403
from lavis.processors import *  # noqa: F401,F403
from lavis.runners import *  # noqa: F401,F403
from lavis.tasks import *  # noqa: F401,F403


DEFAULT_OUTPUT_ROOT = "/home/ywjang/LBA_LAVIS/thesis"
DEFAULT_VQA_INTROSPECT_SOURCE = "/home/ywjang/data/VQA-Introspect/ywjang_sub_qas/vqa_introspect_questioner_subquestions_val_58f05e6001b8.json"
DEFAULT_AOK_VQA_SOURCE = "/home/ywjang/data/AOKVQA/ywjang_sub_qas/aok_vqa_questioner_subquestions_val_58f05e6001b8.json"
DATASET_FOLDER_NAMES = {
    "vqa_introspect": "VQA-Introspect",
    "aok_vqa": "AOK-VQA",
}


class AnswererInferenceDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], vis_processor):
        self.records = records
        self.vis_processor = vis_processor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = Image.open(record["media_path"]).convert("RGB")
        return {
            "image": self.vis_processor(image),
            "source_index": record["source_index"],
            "question_id": record["question_id"],
            "main_question": record["main_question"],
            "media_path": record["media_path"],
            "sub_q": record["sub_q"],
        }

    @staticmethod
    def collater(samples: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "image": torch.stack([sample["image"] for sample in samples], dim=0),
            "source_index": [sample["source_index"] for sample in samples],
            "question_id": [sample["question_id"] for sample in samples],
            "main_question": [sample["main_question"] for sample in samples],
            "media_path": [sample["media_path"] for sample in samples],
            "sub_q": [sample["sub_q"] for sample in samples],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export thesis-format SQ-InstructBLIP sub-QA JSONs.")
    parser.add_argument("--cfg-path", default="instructBLIP_FT_vicuna7b.yaml")
    parser.add_argument("--prompt-path", default="prompts.json")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--answerer-checkpoint", default=None)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["vqa_introspect", "aok_vqa"],
        choices=["vqa_introspect", "aok_vqa"],
    )
    parser.add_argument("--vqa-introspect-source", default=DEFAULT_VQA_INTROSPECT_SOURCE)
    parser.add_argument("--aok-vqa-source", default=DEFAULT_AOK_VQA_SOURCE)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
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


def _default_answerer_checkpoint() -> str:
    latest_local = sorted(
        Path("/home/ywjang/LBA_LAVIS/lavis/ywjang_output_answerer").glob("*/checkpoint_best.pth")
    )
    latest_backup = sorted(
        Path("/home/ywjang/_backup_github/LBA_LAVIS/lavis/ywjang_output_answerer").glob("*/checkpoint_best.pth")
    )
    checkpoint = _first_existing_path(
        [
            latest_local[-1] if latest_local else None,
            latest_backup[-1] if latest_backup else None,
            "/home/ywjang/_backup_github/LBA_LAVIS/lavis/ywjang_output/best_models/20230810124_Answerer/checkpoint_best.pth",
        ]
    )
    if checkpoint is None:
        raise FileNotFoundError("Could not find an Answerer checkpoint. Pass --answerer-checkpoint explicitly.")
    return checkpoint


def _load_answerer_prompt_config(prompt_path: str) -> dict[str, str]:
    prompts = json.loads(Path(prompt_path).read_text())
    return prompts["Answerer"]


def _dataset_source_path(args: argparse.Namespace, dataset_name: str) -> Path:
    if dataset_name == "vqa_introspect":
        return Path(args.vqa_introspect_source)
    if dataset_name == "aok_vqa":
        return Path(args.aok_vqa_source)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _load_questioner_records(source_path: Path) -> list[dict[str, Any]]:
    records = json.loads(source_path.read_text())
    normalized = []
    for source_index, record in enumerate(records):
        normalized.append(
            {
                "source_index": source_index,
                "question_id": str(record["question_id"]),
                "main_question": record["main_question"],
                "media_path": record["image_path"],
                "sub_q": list(record["sub-questions"]),
            }
        )
    return normalized


def _normalize_answer(output: str) -> str:
    if output.endswith("0" * 10):
        return " ".join(output.split()[:-1])
    return output


def _generate_sub_answers_for_batch(
    model,
    images: torch.Tensor,
    sub_questions: list[list[str]],
    answerer_prompt: dict[str, str],
) -> list[list[str]]:
    images = images.to(model.device, non_blocking=True)
    generated = [[] for _ in sub_questions]
    max_sub_questions = max((len(questions) for questions in sub_questions), default=0)

    for step in range(max_sub_questions):
        active_indices = [idx for idx, questions in enumerate(sub_questions) if step < len(questions)]
        if not active_indices:
            continue

        prompts = [answerer_prompt["init_prompt"].format(sub_questions[idx][step]) for idx in active_indices]
        outputs = model.generate(
            {"image": images[active_indices], "prompt": prompts},
        )
        for local_idx, output in enumerate(outputs):
            generated[active_indices[local_idx]].append(_normalize_answer(output))

    return generated


def _gather_rank_records(local_records: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    if not is_dist_avail_and_initialized():
        return local_records

    gathered = [None for _ in range(get_world_size())] if is_main_process() else None
    dist.gather_object(local_records, gathered, dst=0)
    if not is_main_process():
        return None

    merged = []
    for records in gathered:
        merged.extend(records)
    return merged


def _render_readme(final_paths: dict[str, str], source_paths: dict[str, str], answerer_checkpoint: str) -> str:
    lines = [
        "# thesis",
        "",
        "This directory contains thesis-ready SQ-InstructBLIP sub-QA exports for the validation split.",
        "",
        "## Structure",
        "",
        "- `VQA-Introspect/SQ-InstructBLIP/sub_qas.json`",
        "- `AOK-VQA/SQ-InstructBLIP/sub_qas.json`",
        "",
        "## Schema",
        "",
        "Each `sub_qas.json` is a list where each element has a single `question_id` key:",
        "",
        "```json",
        '[{"<qid>": {"main_question": "...", "sub_q": ["..."], "sub_a": ["..."], "media_path": "/abs/path.jpg"}}]',
        "```",
        "",
        "## Provenance",
        "",
        f"- Questioner source for VQA-Introspect: `{source_paths['vqa_introspect']}`",
        f"- Questioner source for AOK-VQA: `{source_paths['aok_vqa']}`",
        f"- Answerer checkpoint: `{answerer_checkpoint}`",
        "",
        "## Final Files",
        "",
        f"- VQA-Introspect: `{final_paths['vqa_introspect']}`",
        f"- AOK-VQA: `{final_paths['aok_vqa']}`",
        "",
    ]
    return "\n".join(lines)


def run_dataset(
    args: argparse.Namespace,
    dataset_name: str,
    model,
    vis_processor,
    answerer_prompt: dict[str, str],
) -> str | None:
    source_path = _dataset_source_path(args, dataset_name)
    records = _load_questioner_records(source_path)
    sharded_records = records[get_rank() :: get_world_size()]
    dataset = AnswererInferenceDataset(sharded_records, vis_processor)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=AnswererInferenceDataset.collater,
    )

    local_records = []
    total_records = len(records)
    total_batches = len(dataloader)
    for batch_index, batch in enumerate(dataloader, start=1):
        sub_answers = _generate_sub_answers_for_batch(
            model=model,
            images=batch["image"],
            sub_questions=batch["sub_q"],
            answerer_prompt=answerer_prompt,
        )

        for source_index, question_id, main_question, media_path, sub_questions, answers in zip(
            batch["source_index"],
            batch["question_id"],
            batch["main_question"],
            batch["media_path"],
            batch["sub_q"],
            sub_answers,
        ):
            local_records.append(
                build_answerer_rank_output_record(
                    source_index=source_index,
                    question_id=question_id,
                    main_question=main_question,
                    media_path=media_path,
                    sub_questions=sub_questions,
                    sub_answers=answers,
                )
            )

        if is_main_process() and (batch_index == 1 or batch_index % args.log_every == 0 or batch_index == total_batches):
            done = min(batch_index * args.batch_size * get_world_size(), total_records)
            print(f"[{dataset_name}] answered {done}/{total_records} examples")

    gathered_records = _gather_rank_records(local_records)
    if gathered_records is None:
        return None

    merged_records = merge_answerer_rank_output_records(gathered_records)
    thesis_payload = to_thesis_sub_qas(merged_records)
    output_dir = Path(args.output_root) / DATASET_FOLDER_NAMES[dataset_name] / "SQ-InstructBLIP"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sub_qas.json"
    output_path.write_text(json.dumps(thesis_payload, indent=2, ensure_ascii=False))
    print(f"Saved {dataset_name} thesis export to {output_path}")
    return str(output_path)


def main() -> None:
    args = parse_args()
    cfg = _build_runtime_cfg(args.cfg_path)
    vis_processor, _ = _build_eval_processors(cfg)
    llm_model = args.llm_model or _default_llm_model_path()
    answerer_checkpoint = args.answerer_checkpoint or _default_answerer_checkpoint()
    prompts = json.loads(Path(args.prompt_path).read_text())
    answerer_prompt = _load_answerer_prompt_config(args.prompt_path)

    answerer, _ = _get_model(
        cfg_path=args.cfg_path,
        finetuned={"Answerer": answerer_checkpoint},
        role="Answerer",
        prompts=prompts,
        llm_model=llm_model,
        build_runtime=False,
    )
    if torch.cuda.is_available():
        answerer = answerer.to(f"cuda:{cfg.run_cfg.gpu}")
    answerer.eval()

    final_paths = {}
    for dataset_name in args.datasets:
        result = run_dataset(
            args=args,
            dataset_name=dataset_name,
            model=answerer,
            vis_processor=vis_processor,
            answerer_prompt=answerer_prompt,
        )
        if result is not None:
            final_paths[dataset_name] = result
        if is_dist_avail_and_initialized():
            dist.barrier()

    if is_main_process():
        source_paths = {
            "vqa_introspect": str(_dataset_source_path(args, "vqa_introspect")),
            "aok_vqa": str(_dataset_source_path(args, "aok_vqa")),
        }
        readme_path = Path(args.output_root) / "readme.md"
        readme_path.parent.mkdir(parents=True, exist_ok=True)
        readme_path.write_text(_render_readme(final_paths, source_paths, answerer_checkpoint))
        print(json.dumps(final_paths, indent=2))
        print(str(readme_path))

    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
