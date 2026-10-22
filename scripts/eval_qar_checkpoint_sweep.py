#!/usr/bin/env python3
import argparse
import csv
import gc
import json
import os
import re
import string
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import easydict
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.datasets.builders import *  # noqa: F401,F403
from lavis.models import *  # noqa: F401,F403
from lavis.processors import *  # noqa: F401,F403
from scripts.multigpu_eval_utils import merge_json_shards, parse_device_list, run_parallel_commands, shard_indices, worker_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Small QAR checkpoint sweep on a uniform eval subset")
    parser.add_argument("--cfg-path", default="configs/sq_instructblip/reasoner.yaml")
    parser.add_argument("--questioner-dir", default="lavis/ywjang_output_sq_fraction/questioner_small/20260528180")
    parser.add_argument("--reasoner-dir", default="lavis/ywjang_output_sq_fraction/reasoner_small/20260528232")
    parser.add_argument("--answerer-checkpoint", default="/model/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth")
    parser.add_argument("--llm-model", default="/model/lmsys/vicuna-7b-v1.1")
    parser.add_argument("--ann-val", default="/home/ywjang/LBA_LAVIS/data/VQA-Introspect/VQAIntrospect_valv1.0.json")
    parser.add_argument("--image-root", default="/data/coco/images")
    parser.add_argument("--output-dir", default="lavis/ywjang_output_sq_fraction/qar_small_uniform10_sweep")
    parser.add_argument("--eval-sample-fraction", type=float, default=0.1)
    parser.add_argument("--eval-sample-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-rank", type=int, default=None)
    parser.add_argument("--checkpoint-start", type=int, default=0)
    parser.add_argument("--checkpoint-end", type=int, default=9)
    parser.add_argument("--summary-prefix", default="summary")
    parser.add_argument("--naive-results", default=None, help="results.json from scripts/eval_instructblip_naive.py")
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--num-beams", type=int, default=5)
    return parser.parse_args()


def config_options(args, *, prompt_type, checkpoint, role, arch="blip2_vicuna_instruct"):
    return [
        "model.load_finetuned=True",
        f"model.finetuned={checkpoint}",
        f"model.arch={arch}",
        f"model.role={role}",
        f"model.llm_model={args.llm_model}",
        f"datasets.vqa_introspect_qar_caption.prompt_type={prompt_type}",
        f"datasets.vqa_introspect_qar_caption.build_info.annotations.val.storage=[{args.ann_val}]",
        f"datasets.vqa_introspect_qar_caption.build_info.annotations.test.storage=[{args.ann_val}]",
        f"datasets.vqa_introspect_qar_caption.build_info.images.storage={args.image_root}",
        f"datasets.vqa_introspect_qar_caption.eval_sample_fraction={args.eval_sample_fraction}",
        f"datasets.vqa_introspect_qar_caption.eval_sample_seed={args.eval_sample_seed}",
        "run.valid_splits=[]",
        "run.test_splits=[val]",
        "run.batch_size_eval=1",
        "run.num_workers=0",
        f"run.max_len={args.max_len}",
        f"run.min_len={args.min_len}",
        f"run.num_beams={args.num_beams}",
    ]


def build_cfg(args, *, prompt_type="QAR", checkpoint=None, role="Reasoner"):
    checkpoint = checkpoint or args.answerer_checkpoint
    cfg_args = easydict.EasyDict(
        {
            "cfg_path": args.cfg_path,
            "options": config_options(
                args,
                prompt_type=prompt_type,
                checkpoint=checkpoint,
                role=role,
            ),
        }
    )
    return Config(cfg_args)


def build_eval_dataset(args):
    cfg = build_cfg(args, prompt_type="QAR", checkpoint=args.answerer_checkpoint, role="Reasoner")
    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    return datasets["vqa_introspect_qar_caption"]["val"]


def load_model(args, *, checkpoint, role):
    cfg = build_cfg(args, prompt_type="QAR", checkpoint=checkpoint, role=role)
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    model.eval()
    model.to(args.device)
    return model


def move_sample_to_device(sample, device):
    out = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def cleanup_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def normalize_answer(answer):
    answer = str(answer).lower().strip()
    answer = answer.translate(str.maketrans("", "", string.punctuation))
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    answer = re.sub(r"\s+", " ", answer).strip()
    return answer


QUESTION_STARTERS = {
    "am",
    "are",
    "can",
    "could",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "how",
    "is",
    "may",
    "might",
    "must",
    "shall",
    "should",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "will",
    "would",
}


def is_question_form(output):
    text = str(output).strip()
    if not text:
        return False
    if text.endswith("?"):
        return True
    match = re.match(r"[A-Za-z']+", text)
    return bool(match and match.group(0).lower() in QUESTION_STARTERS)


TARGET_CONFUSION_SUMMARY_KEYS = [
    "questioner_naive_main_answer_count",
    "questioner_naive_main_answer_rate",
    "questioner_not_question_count",
    "questioner_not_question_rate",
    "reasoner_sub_answer_instead_count",
    "reasoner_sub_answer_instead_rate",
    "reasoner_question_form_examples",
    "reasoner_non_question_excluded_count",
    "reasoner_eligible_examples",
    "reasoner_ambiguous_count",
]


def load_naive_results(path):
    if not path:
        return None
    with open(path) as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"naive results must be a JSON list: {path}")
    return rows


def _row_eval_index(row, row_name):
    if "eval_index" not in row:
        raise ValueError(f"{row_name} row missing eval_index")
    return int(row["eval_index"])


def _required_value(row, key, row_name, eval_index):
    if key not in row:
        raise ValueError(f"{row_name} row eval_index={eval_index} missing {key}")
    return row[key]


def _questioner_output(row, eval_index):
    if "questioner_sub_question" in row:
        return row["questioner_sub_question"]
    if "sub_question" in row:
        return row["sub_question"]
    raise ValueError(f"QAR row eval_index={eval_index} missing sub_question")


def _index_naive_rows(naive_rows):
    if naive_rows is None:
        return None
    by_eval_index = {}
    for row in naive_rows:
        eval_index = _row_eval_index(row, "naive")
        _required_value(row, "pred_answer", "naive", eval_index)
        if eval_index in by_eval_index:
            raise ValueError(f"duplicate naive eval_index {eval_index}")
        by_eval_index[eval_index] = row
    return by_eval_index


def apply_target_confusion_metrics(rows, naive_rows=None):
    naive_by_eval_index = _index_naive_rows(naive_rows)
    questioner_count = 0
    questioner_not_question_count = 0
    reasoner_instead_count = 0
    reasoner_question_form_count = 0
    reasoner_non_question_excluded_count = 0
    reasoner_eligible = 0
    reasoner_ambiguous_count = 0

    for row in rows:
        eval_index = _row_eval_index(row, "QAR")
        questioner_output = _questioner_output(row, eval_index)
        output_is_question = int(is_question_form(questioner_output))
        row["questioner_output_is_question"] = output_is_question
        row["questioner_not_question"] = int(not output_is_question)
        questioner_not_question_count += row["questioner_not_question"]

        if naive_by_eval_index is None:
            row["questioner_pred_naive_main_answer"] = None
        else:
            if eval_index not in naive_by_eval_index:
                raise ValueError(f"missing naive eval_index {eval_index}")
            naive_pred = _required_value(
                naive_by_eval_index[eval_index], "pred_answer", "naive", eval_index
            )
            questioner_flag = int(
                normalize_answer(questioner_output) == normalize_answer(naive_pred)
            )
            row["questioner_pred_naive_main_answer"] = questioner_flag
            questioner_count += questioner_flag

        pred_norm = normalize_answer(_required_value(row, "pred_answer", "QAR", eval_index))
        sub_norm = normalize_answer(_required_value(row, "sub_answer", "QAR", eval_index))
        gt_norm = normalize_answer(_required_value(row, "gt_answer", "QAR", eval_index))

        ambiguous = int(gt_norm == sub_norm)
        row["reasoner_target_ambiguous"] = ambiguous
        row["reasoner_question_form_candidate"] = output_is_question
        if not output_is_question:
            reasoner_non_question_excluded_count += 1
            row["reasoner_pred_sub_answer_instead_of_main"] = 0
            continue

        reasoner_question_form_count += 1
        if ambiguous:
            reasoner_ambiguous_count += 1
            row["reasoner_pred_sub_answer_instead_of_main"] = 0
            continue

        reasoner_eligible += 1
        instead = int(pred_norm == sub_norm and pred_norm != gt_norm)
        row["reasoner_pred_sub_answer_instead_of_main"] = instead
        reasoner_instead_count += instead

    has_naive = naive_by_eval_index is not None
    return {
        "questioner_naive_main_answer_count": questioner_count if has_naive else None,
        "questioner_naive_main_answer_rate": (
            questioner_count / len(rows) if has_naive and rows else (0.0 if has_naive else None)
        ),
        "questioner_not_question_count": questioner_not_question_count,
        "questioner_not_question_rate": questioner_not_question_count / len(rows) if rows else 0.0,
        "reasoner_sub_answer_instead_count": reasoner_instead_count,
        "reasoner_sub_answer_instead_rate": (
            reasoner_instead_count / reasoner_eligible if reasoner_eligible else 0.0
        ),
        "reasoner_question_form_examples": reasoner_question_form_count,
        "reasoner_non_question_excluded_count": reasoner_non_question_excluded_count,
        "reasoner_eligible_examples": reasoner_eligible,
        "reasoner_ambiguous_count": reasoner_ambiguous_count,
    }


def eval_dataset_for_rank(args):
    dataset = build_eval_dataset(args)
    indices = list(range(len(dataset)))
    if args.shard_rank is not None:
        indices = shard_indices(len(dataset), args.num_shards, args.shard_rank)
        dataset_view = Subset(dataset, indices)
    else:
        dataset_view = dataset
    return dataset, dataset_view, indices


def generate_questioner(args, base_dataset, dataset, indices, q_ckpt, prompts, ckpt_idx):
    model = load_model(args, checkpoint=q_ckpt, role="Questioner")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=base_dataset.collater)
    q_prompt = prompts["Questioner_MultipleSubQ"]["init_prompt"]
    rows = []
    for offset, sample in enumerate(tqdm(loader, desc=f"ckpt {ckpt_idx} questioner", dynamic_ncols=True, disable=args.shard_rank not in (None, 0))):
        sample = move_sample_to_device(sample, args.device)
        main_question = sample["text_input"][0]
        sample["prompt"] = [q_prompt.format(main_question)]
        output = model.generate(
            sample,
            use_nucleus_sampling=False,
            num_beams=args.num_beams,
            max_length=args.max_len,
            min_length=args.min_len,
        )[0]
        rows.append(
            {
                "eval_index": int(indices[offset]),
                "image_id": int(sample["image_id"][0]),
                "main_question_id": int(sample["main_question_id"][0]),
                "main_question": main_question,
                "gt_answer": sample["text_output"][0],
                "sub_question": output,
            }
        )
    cleanup_model(model)
    return rows


def generate_answerer(args, base_dataset, dataset, rows, prompts, ckpt_idx):
    model = load_model(args, checkpoint=args.answerer_checkpoint, role="Answerer")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=base_dataset.collater)
    a_prompt = prompts["Answerer"]["init_prompt"]
    for row, sample in tqdm(zip(rows, loader), total=len(rows), desc=f"ckpt {ckpt_idx} answerer", dynamic_ncols=True, disable=args.shard_rank not in (None, 0)):
        sample = move_sample_to_device(sample, args.device)
        sample["prompt"] = [a_prompt.format(row["sub_question"])]
        output = model.generate(
            sample,
            use_nucleus_sampling=False,
            num_beams=args.num_beams,
            max_length=args.max_len,
            min_length=args.min_len,
        )[0]
        row["sub_answer"] = output
    cleanup_model(model)
    return rows


def generate_reasoner(args, base_dataset, dataset, rows, r_ckpt, prompts, ckpt_idx):
    model = load_model(args, checkpoint=r_ckpt, role="Reasoner")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=base_dataset.collater)
    r_prompt = prompts["Reasoner"]
    correct = 0
    for row, sample in tqdm(zip(rows, loader), total=len(rows), desc=f"ckpt {ckpt_idx} reasoner", dynamic_ncols=True, disable=args.shard_rank not in (None, 0)):
        sample = move_sample_to_device(sample, args.device)
        main_question = row["main_question"]
        prompt = r_prompt["init_prompt"]
        if prompt.count("{}") == 1:
            prompt = prompt.format(main_question)
        if r_prompt["pair_prompt"].count("{}") == 4:
            prompt += r_prompt["pair_prompt"].format(1, row["sub_question"], 1, row["sub_answer"])
        else:
            prompt += r_prompt["pair_prompt"].format(row["sub_question"], row["sub_answer"])
        prompt += r_prompt["final_prompt"].format(main_question)
        sample["prompt"] = [prompt]
        output = model.generate(
            sample,
            use_nucleus_sampling=False,
            num_beams=args.num_beams,
            max_length=args.max_len,
            min_length=args.min_len,
        )[0]
        row["pred_answer"] = output
        row["normalized_pred"] = normalize_answer(output)
        row["normalized_gt"] = normalize_answer(row["gt_answer"])
        row["exact_match"] = int(row["normalized_pred"] == row["normalized_gt"])
        correct += row["exact_match"]
    cleanup_model(model)
    return rows, correct / max(len(rows), 1)


def save_rows(path, rows):
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)


def load_rows_if_complete(path, expected_len, required_keys):
    if not path.exists():
        return None
    rows = json.load(open(path))
    if len(rows) != expected_len:
        print(f"ignore {path}: expected {expected_len} rows, found {len(rows)}")
        return None
    missing = [key for key in required_keys if any(key not in row for row in rows)]
    if missing:
        print(f"ignore {path}: missing keys {missing}")
        return None
    return rows


def write_summary(out_dir, prefix, summary):
    with open(out_dir / f"{prefix}.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / f"{prefix}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def summarize_checkpoint(out_dir, prefix, summary, idx, q_ckpt, r_ckpt, result_file, acc, naive_rows=None):
    with open(result_file) as f:
        rows = json.load(f)
    target_confusion = apply_target_confusion_metrics(rows, naive_rows=naive_rows)
    save_rows(result_file, rows)

    item = {
        "checkpoint": idx,
        "questioner_checkpoint": str(q_ckpt),
        "reasoner_checkpoint": str(r_ckpt),
        "num_examples": len(rows),
        "accuracy": acc,
    }
    item.update(target_confusion)
    summary.append(item)
    write_summary(out_dir, prefix, summary)


def run_single(args):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = json.load(open("prompts.json"))
    naive_rows = load_naive_results(args.naive_results)
    base_dataset, dataset, indices = eval_dataset_for_rank(args)
    print(f"uniform eval subset size: {len(base_dataset)}")
    if args.shard_rank is not None:
        print(f"QAR shard {args.shard_rank}/{args.num_shards}: {len(indices)} examples on {args.device}")

    summary = []
    for idx in range(args.checkpoint_start, args.checkpoint_end + 1):
        q_ckpt = Path(args.questioner_dir) / f"checkpoint_{idx}.pth"
        r_ckpt = Path(args.reasoner_dir) / f"checkpoint_{idx}.pth"
        if not q_ckpt.is_file() or not r_ckpt.is_file():
            print(f"skip checkpoint_{idx}: missing {q_ckpt} or {r_ckpt}")
            continue
        result_file = out_dir / f"checkpoint_{idx}_results.json"
        marker = out_dir / f"checkpoint_{idx}.done"
        if marker.exists() and result_file.exists():
            rows = json.load(open(result_file))
            acc = sum(int(r.get("exact_match", 0)) for r in rows) / max(len(rows), 1)
            print(f"skip checkpoint_{idx}: existing accuracy={acc:.6f}")
        else:
            print(f"START checkpoint_{idx}")
            question_file = out_dir / f"checkpoint_{idx}_questions.json"
            subqa_file = out_dir / f"checkpoint_{idx}_subqa.json"

            rows = load_rows_if_complete(
                question_file,
                len(dataset),
                ["eval_index", "image_id", "main_question_id", "main_question", "gt_answer", "sub_question"],
            )
            if rows is None:
                rows = generate_questioner(args, base_dataset, dataset, indices, str(q_ckpt), prompts, idx)
                save_rows(question_file, rows)
            else:
                print(f"resume checkpoint_{idx}: using {question_file}")

            resumed_subqa = load_rows_if_complete(
                subqa_file,
                len(dataset),
                ["eval_index", "image_id", "main_question_id", "main_question", "gt_answer", "sub_question", "sub_answer"],
            )
            if resumed_subqa is None:
                rows = generate_answerer(args, base_dataset, dataset, rows, prompts, idx)
                save_rows(subqa_file, rows)
            else:
                print(f"resume checkpoint_{idx}: using {subqa_file}")
                rows = resumed_subqa

            rows, acc = generate_reasoner(args, base_dataset, dataset, rows, str(r_ckpt), prompts, idx)
            save_rows(result_file, rows)
            marker.write_text("done\n")
            print(f"DONE checkpoint_{idx}: accuracy={acc:.6f}")
        summarize_checkpoint(out_dir, args.summary_prefix, summary, idx, q_ckpt, r_ckpt, result_file, acc, naive_rows=naive_rows)

    print(json.dumps(summary, indent=2))


def worker_cmd(args, idx, device, rank, shard_dir):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--cfg-path",
        args.cfg_path,
        "--questioner-dir",
        args.questioner_dir,
        "--reasoner-dir",
        args.reasoner_dir,
        "--answerer-checkpoint",
        args.answerer_checkpoint,
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
        "--checkpoint-start",
        str(idx),
        "--checkpoint-end",
        str(idx),
        "--summary-prefix",
        args.summary_prefix,
        "--max-len",
        str(args.max_len),
        "--min-len",
        str(args.min_len),
        "--num-beams",
        str(args.num_beams),
    ]
    if args.naive_results:
        command += ["--naive-results", args.naive_results]
    return command


def run_multi_device(args):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    naive_rows = load_naive_results(args.naive_results)

    for idx in range(args.checkpoint_start, args.checkpoint_end + 1):
        q_ckpt = Path(args.questioner_dir) / f"checkpoint_{idx}.pth"
        r_ckpt = Path(args.reasoner_dir) / f"checkpoint_{idx}.pth"
        if not q_ckpt.is_file() or not r_ckpt.is_file():
            print(f"skip checkpoint_{idx}: missing {q_ckpt} or {r_ckpt}")
            continue

        result_file = out_dir / f"checkpoint_{idx}_results.json"
        marker = out_dir / f"checkpoint_{idx}.done"
        if marker.exists() and result_file.exists():
            rows = json.load(open(result_file))
            acc = sum(int(r.get("exact_match", 0)) for r in rows) / max(len(rows), 1)
            print(f"skip checkpoint_{idx}: existing accuracy={acc:.6f}")
        else:
            shard_root = out_dir / "shards" / f"checkpoint_{idx}"
            commands = []
            shard_result_files = []
            for rank, device in enumerate(args.devices):
                shard_dir = shard_root / f"rank_{rank}"
                shard_result_files.append(shard_dir / f"checkpoint_{idx}_results.json")
                commands.append(worker_cmd(args, idx, device, rank, shard_dir))
            run_parallel_commands(commands)
            rows = merge_json_shards(shard_result_files)
            save_rows(result_file, rows)
            marker.write_text("done\n")
            acc = sum(int(r.get("exact_match", 0)) for r in rows) / max(len(rows), 1)
            print(f"DONE checkpoint_{idx}: accuracy={acc:.6f}")
        summarize_checkpoint(out_dir, args.summary_prefix, summary, idx, q_ckpt, r_ckpt, result_file, acc, naive_rows=naive_rows)

    print(json.dumps(summary, indent=2))


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
