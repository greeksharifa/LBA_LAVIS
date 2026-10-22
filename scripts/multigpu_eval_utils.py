#!/usr/bin/env python3
import json
import logging
import os
from contextlib import contextmanager, redirect_stdout
import subprocess
import sys
from pathlib import Path


@contextmanager
def worker_logging(args):
    if args.shard_rank in (None, 0):
        yield
        return
    previous_level = logging.root.manager.disable
    try:
        logging.disable(logging.INFO)
        with open(os.devnull, "w") as sink, redirect_stdout(sink):
            yield
    finally:
        logging.disable(previous_level)


def parse_device_list(primary_device, devices):
    if devices:
        return list(devices)
    return [primary_device]


def shard_indices(total, num_shards, shard_rank):
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_rank < 0 or shard_rank >= num_shards:
        raise ValueError(f"shard_rank must be in [0, {num_shards}), got {shard_rank}")
    return [idx for idx in range(total) if idx % num_shards == shard_rank]


def merge_json_shards(paths):
    rows = []
    for path in paths:
        path = Path(path)
        shard_rows = json.load(open(path))
        rows.extend(shard_rows)
    rows.sort(key=lambda row: int(row["eval_index"]))
    return rows


def run_parallel_commands(commands):
    procs = []
    try:
        for cmd in commands:
            print("$ " + " ".join(str(part) for part in cmd), flush=True)
            procs.append(subprocess.Popen([str(part) for part in cmd]))
        failures = []
        for proc, cmd in zip(procs, commands):
            status = proc.wait()
            if status != 0:
                failures.append((status, cmd))
        if failures:
            for proc in procs:
                if proc.poll() is None:
                    proc.terminate()
            status, cmd = failures[0]
            raise subprocess.CalledProcessError(status, [str(part) for part in cmd])
    except BaseException:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        raise


def build_worker_base_cmd(script_path, args, output_dir, checkpoint_start=None, checkpoint_end=None):
    cmd = [
        sys.executable,
        str(script_path),
        "--cfg-path",
        args.cfg_path,
        "--llm-model",
        args.llm_model,
        "--ann-val",
        args.ann_val,
        "--image-root",
        args.image_root,
        "--output-dir",
        str(output_dir),
        "--eval-sample-fraction",
        str(args.eval_sample_fraction),
        "--eval-sample-seed",
        str(args.eval_sample_seed),
        "--max-len",
        str(args.max_len),
        "--min-len",
        str(args.min_len),
        "--num-beams",
        str(args.num_beams),
    ]
    if checkpoint_start is not None:
        cmd += ["--checkpoint-start", str(checkpoint_start)]
    if checkpoint_end is not None:
        cmd += ["--checkpoint-end", str(checkpoint_end)]
    return cmd
