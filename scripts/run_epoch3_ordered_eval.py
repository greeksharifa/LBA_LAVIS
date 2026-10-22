#!/usr/bin/env python3
import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path

from scripts.multigpu_eval_utils import parse_device_list

FRACTIONS = {"small": "0.125", "medium": "0.25", "large": "0.5", "full": "1.0"}


def parse_devices(primary_device, devices):
    return parse_device_list(primary_device, devices)


def output_root_path(out_root):
    path = Path(out_root)
    if path.is_absolute() or (path.parts and path.parts[0] == "lavis"):
        return path
    return Path("lavis") / path


def latest_checkpoint_dir(root, role, version):
    base = Path(root) / f"{role}_{version}"
    if not base.is_dir():
        raise FileNotFoundError(f"missing checkpoint root: {base}")
    dirs = sorted(
        [path for path in base.iterdir() if path.is_dir()],
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    for directory in dirs:
        if (directory / "checkpoint_2.pth").is_file():
            return directory
    raise FileNotFoundError(f"missing checkpoint_2.pth under {base}")


def checkpoint_dir_for_run(root, role, version, dry_run=False):
    try:
        return latest_checkpoint_dir(root, role, version)
    except FileNotFoundError:
        if dry_run:
            return Path(root) / f"{role}_{version}" / "DRY_RUN_RUN_DIR"
        raise


def run_cmd(cmd, dry_run=False):
    print("$ " + " ".join(shlex.quote(str(part)) for part in cmd), flush=True)
    if not dry_run:
        subprocess.run([str(part) for part in cmd], check=True)


def append_device_args(cmd, args):
    cmd += ["--device", args.device]
    if len(args.devices) > 1:
        cmd += ["--devices", *args.devices]
    return cmd


def run_naive(args, ordered_dir):
    naive_dir = ordered_dir / "01_naive_base"
    cmd = [
        args.python_executable,
        "scripts/eval_instructblip_naive.py",
        "--cfg-path",
        args.cfg_path,
        "--model-checkpoint",
        args.base_checkpoint,
        "--llm-model",
        args.llm_model,
        "--ann-val",
        args.ann_val,
        "--image-root",
        args.image_root,
        "--output-dir",
        naive_dir,
        "--eval-sample-fraction",
        str(args.eval_sample_fraction),
        "--eval-sample-seed",
        str(args.eval_sample_seed),
        "--summary-prefix",
        "summary",
        "--max-len",
        str(args.max_len),
        "--min-len",
        str(args.min_len),
        "--num-beams",
        str(args.num_beams),
    ]
    run_cmd(append_device_args(cmd, args), dry_run=args.dry_run)


def run_qar(args, root, ordered_dir, version, order):
    q_dir = checkpoint_dir_for_run(root, "questioner", version, dry_run=args.dry_run)
    r_dir = checkpoint_dir_for_run(root, "reasoner", version, dry_run=args.dry_run)
    eval_dir = ordered_dir / f"{order:02d}_{version}_ckpt2"
    cmd = [
        args.python_executable,
        "scripts/eval_qar_checkpoint_sweep.py",
        "--cfg-path",
        args.cfg_path,
        "--questioner-dir",
        q_dir,
        "--reasoner-dir",
        r_dir,
        "--answerer-checkpoint",
        args.answerer_checkpoint,
        "--llm-model",
        args.llm_model,
        "--ann-val",
        args.ann_val,
        "--image-root",
        args.image_root,
        "--output-dir",
        eval_dir,
        "--eval-sample-fraction",
        str(args.eval_sample_fraction),
        "--eval-sample-seed",
        str(args.eval_sample_seed),
        "--checkpoint-start",
        "2",
        "--checkpoint-end",
        "2",
        "--summary-prefix",
        "summary_ckpt2",
        "--naive-results",
        ordered_dir / "01_naive_base" / "results.json",
        "--max-len",
        str(args.max_len),
        "--min-len",
        str(args.min_len),
        "--num-beams",
        str(args.num_beams),
    ]
    run_cmd(append_device_args(cmd, args), dry_run=args.dry_run)


def read_single_qar_summary(path):
    summary = json.load(open(path))
    if len(summary) != 1 or int(summary[0]["checkpoint"]) != 2:
        raise ValueError(f"unexpected QAR summary contents in {path}: {summary}")
    return summary[0]


def copy_summary_extras(row, source):
    for key, value in source.items():
        if key not in row:
            row[key] = value


def ordered_fieldnames(rows):
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def aggregate_ordered_summary(root, ordered_dir, base_checkpoint, versions=("small", "medium", "large")):
    ordered_dir = Path(ordered_dir)
    rows = []

    naive_dir = ordered_dir / "01_naive_base"
    naive = json.load(open(naive_dir / "summary.json"))
    naive_row = {
        "order": 1,
        "setting": "naive",
        "eval_type": "naive_instructblip",
        "train_fraction": "0",
        "checkpoint": naive.get("checkpoint", "base"),
        "num_examples": naive["num_examples"],
        "accuracy": naive["accuracy"],
        "model_checkpoint": naive.get("model_checkpoint", base_checkpoint),
        "questioner_checkpoint": "",
        "reasoner_checkpoint": "",
        "output_dir": str(naive_dir),
    }
    copy_summary_extras(naive_row, naive)
    rows.append(naive_row)

    for offset, version in enumerate(versions, start=2):
        eval_dir = ordered_dir / f"{offset:02d}_{version}_ckpt2"
        item = read_single_qar_summary(eval_dir / "summary_ckpt2.json")
        qar_row = {
            "order": offset,
            "setting": version,
            "eval_type": "sq_instructblip_qar",
            "train_fraction": FRACTIONS.get(version, ""),
            "checkpoint": int(item["checkpoint"]),
            "num_examples": item["num_examples"],
            "accuracy": item["accuracy"],
            "model_checkpoint": "",
            "questioner_checkpoint": item["questioner_checkpoint"],
            "reasoner_checkpoint": item["reasoner_checkpoint"],
            "output_dir": str(eval_dir),
        }
        copy_summary_extras(qar_row, item)
        rows.append(qar_row)

    suffix = "_".join(["naive", *versions])
    json_path = ordered_dir / f"summary_ordered_{suffix}.json"
    csv_path = ordered_dir / f"summary_ordered_{suffix}.csv"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {csv_path}", flush=True)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Run ordered epoch-3 eval: naive, then QAR versions")
    parser.add_argument("--out-root", default="ywjang_output_sq_epoch3_fraction_tmux")
    parser.add_argument("--ordered-subdir", default="ordered_eval")
    parser.add_argument("--cfg-path", default="configs/sq_instructblip/reasoner.yaml")
    parser.add_argument("--base-checkpoint", default="/model/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth")
    parser.add_argument("--answerer-checkpoint", default=None)
    parser.add_argument("--llm-model", default="/model/lmsys/vicuna-7b-v1.1")
    parser.add_argument("--ann-val", default="/home/ywjang/data/VQA-Introspect/VQAIntrospect_valv1.0.json")
    parser.add_argument("--image-root", default="/data/coco/images")
    parser.add_argument("--eval-sample-fraction", type=float, default=0.1)
    parser.add_argument("--eval-sample-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--versions", nargs="+", default=["small", "medium", "large"])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.answerer_checkpoint is None:
        args.answerer_checkpoint = args.base_checkpoint
    args.devices = parse_devices(args.device, args.devices)

    root = output_root_path(args.out_root)
    ordered_dir = root / args.ordered_subdir
    ordered_dir.mkdir(parents=True, exist_ok=True)

    run_naive(args, ordered_dir)
    for order, version in enumerate(args.versions, start=2):
        run_qar(args, root, ordered_dir, version, order)

    if not args.dry_run:
        rows = aggregate_ordered_summary(
            root=root,
            ordered_dir=ordered_dir,
            base_checkpoint=args.base_checkpoint,
            versions=tuple(args.versions),
        )
        print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
