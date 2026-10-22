#!/usr/bin/env python3
import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from scripts.eval_qar_checkpoint_sweep import (
    build_eval_dataset,
    cleanup_model,
    load_model,
    move_sample_to_device,
    normalize_answer,
)
from scripts.multigpu_eval_utils import merge_json_shards, parse_device_list, run_parallel_commands, shard_indices, worker_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate base InstructBLIP directly on VQA-Introspect")
    parser.add_argument("--cfg-path", default="configs/sq_instructblip/reasoner.yaml")
    parser.add_argument("--model-checkpoint", default="/model/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth")
    parser.add_argument("--llm-model", default="/model/lmsys/vicuna-7b-v1.1")
    parser.add_argument("--ann-val", default="/home/ywjang/data/VQA-Introspect/VQAIntrospect_valv1.0.json")
    parser.add_argument("--image-root", default="/data/coco/images")
    parser.add_argument("--output-dir", default="lavis/ywjang_output_sq_epoch3_fraction/ordered_eval/01_naive_base")
    parser.add_argument("--eval-sample-fraction", type=float, default=0.1)
    parser.add_argument("--eval-sample-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-rank", type=int, default=None)
    parser.add_argument("--summary-prefix", default="summary")
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--num-beams", type=int, default=5)
    return parser.parse_args()


def write_summary(out_dir, prefix, summary):
    with open(out_dir / f"{prefix}.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / f"{prefix}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def write_outputs(out_dir, result_file, marker, rows, args):
    accuracy = sum(int(row.get("exact_match", 0)) for row in rows) / max(len(rows), 1)
    with open(result_file, "w") as f:
        json.dump(rows, f, indent=2)
    marker.write_text("done\n")
    summary = {
        "setting": "naive",
        "checkpoint": "base",
        "num_examples": len(rows),
        "accuracy": accuracy,
        "model_checkpoint": args.model_checkpoint,
    }
    write_summary(out_dir, args.summary_prefix, summary)
    print(json.dumps(summary, indent=2))
    return summary


def eval_dataset_for_rank(args):
    dataset = build_eval_dataset(args)
    indices = list(range(len(dataset)))
    if args.shard_rank is not None:
        indices = shard_indices(len(dataset), args.num_shards, args.shard_rank)
        dataset_view = Subset(dataset, indices)
    else:
        dataset_view = dataset
    return dataset, dataset_view, indices


def run_single(args):
    args.answerer_checkpoint = args.model_checkpoint

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_file = out_dir / "results.json"
    marker = out_dir / "done"

    if marker.exists() and result_file.exists():
        rows = json.load(open(result_file))
        accuracy = sum(int(row.get("exact_match", 0)) for row in rows) / max(len(rows), 1)
        print(f"skip naive: existing accuracy={accuracy:.6f}")
    else:
        base_dataset, dataset, indices = eval_dataset_for_rank(args)
        print(f"uniform eval subset size: {len(base_dataset)}")
        if args.shard_rank is not None:
            print(f"naive shard {args.shard_rank}/{args.num_shards}: {len(indices)} examples on {args.device}")
        model = load_model(args, checkpoint=args.model_checkpoint, role="Answerer")
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=base_dataset.collater)

        rows = []
        correct = 0
        for offset, sample in enumerate(tqdm(loader, desc="naive instructblip", dynamic_ncols=True, disable=args.shard_rank not in (None, 0))):
            sample = move_sample_to_device(sample, args.device)
            main_question = sample["text_input"][0]
            sample["prompt"] = [main_question]
            with torch.no_grad():
                output = model.generate(
                    sample,
                    use_nucleus_sampling=False,
                    num_beams=args.num_beams,
                    max_length=args.max_len,
                    min_length=args.min_len,
                )[0]
            normalized_pred = normalize_answer(output)
            normalized_gt = normalize_answer(sample["text_output"][0])
            exact_match = int(normalized_pred == normalized_gt)
            correct += exact_match
            rows.append(
                {
                    "eval_index": int(indices[offset]),
                    "image_id": int(sample["image_id"][0]),
                    "main_question_id": int(sample["main_question_id"][0]),
                    "main_question": main_question,
                    "gt_answer": sample["text_output"][0],
                    "pred_answer": output,
                    "normalized_pred": normalized_pred,
                    "normalized_gt": normalized_gt,
                    "exact_match": exact_match,
                }
            )

        accuracy = correct / max(len(rows), 1)
        with open(result_file, "w") as f:
            json.dump(rows, f, indent=2)
        marker.write_text("done\n")
        cleanup_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"DONE naive: accuracy={accuracy:.6f}")

    summary = {
        "setting": "naive",
        "checkpoint": "base",
        "num_examples": len(rows),
        "accuracy": accuracy,
        "model_checkpoint": args.model_checkpoint,
    }
    write_summary(out_dir, args.summary_prefix, summary)
    print(json.dumps(summary, indent=2))


def worker_cmd(args, device, rank, shard_dir):
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--cfg-path",
        args.cfg_path,
        "--model-checkpoint",
        args.model_checkpoint,
        "--llm-model",
        args.llm_model,
        "--ann-val",
        args.ann_val,
        "--image-root",
        args.image_root,
        "--output-dir",
        str(shard_dir),
        "--eval-sample-fraction",
        str(args.eval_sample_fraction),
        "--eval-sample-seed",
        str(args.eval_sample_seed),
        "--device",
        device,
        "--num-shards",
        str(len(args.devices)),
        "--shard-rank",
        str(rank),
        "--summary-prefix",
        args.summary_prefix,
        "--max-len",
        str(args.max_len),
        "--min-len",
        str(args.min_len),
        "--num-beams",
        str(args.num_beams),
    ]


def run_multi_device(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_file = out_dir / "results.json"
    marker = out_dir / "done"

    if marker.exists() and result_file.exists():
        rows = json.load(open(result_file))
        print("skip naive multi-gpu: existing results")
        write_outputs(out_dir, result_file, marker, rows, args)
        return

    shard_root = out_dir / "shards"
    commands = []
    shard_result_files = []
    for rank, device in enumerate(args.devices):
        shard_dir = shard_root / f"rank_{rank}"
        shard_result_files.append(shard_dir / "results.json")
        commands.append(worker_cmd(args, device, rank, shard_dir))
    run_parallel_commands(commands)

    rows = merge_json_shards(shard_result_files)
    write_outputs(out_dir, result_file, marker, rows, args)


def main():
    args = parse_args()
    args.devices = parse_device_list(args.device, args.devices)
    if len(args.devices) > 1 and args.shard_rank is None:
        run_multi_device(args)
    else:
        with worker_logging(args):
            run_single(args)


if __name__ == "__main__":
    main()
