"""Inference-stage orchestration and artifact persistence."""

import json
import os
import tempfile
from collections import Counter, OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Callable, Optional

from prompt.postprocess import format_vllm_outputs
from prompt.prompts import (
    get_base_prompt,
    get_refined_prompt,
    get_suba_prompt,
    get_subq_prompt,
)
from util.artifacts import (
    MANIFEST_FILENAME,
    create_manifest,
    load_manifest,
    mark_stage_complete,
    mark_stage_started,
    validate_manifest,
    write_manifest,
)
from util.logger import get_logger
from util.path import get_output_dir, get_output_filename
from util.utils import IndexSampler, data_print, json_default


STAGE_ORDER = ("subq", "suba", "base", "refined")
OUTPUT_KEYS = {
    "subq": ("subq_list", "conf_subq"),
    "suba": ("suba_list", "conf_suba"),
    "base": ("base_answer", "conf_base"),
    "refined": ("refined_answer_list", "conf_refined"),
}
CONFIDENCE_KEYS = {"seq_ppl", "token_min_prob"}


def write_json_atomic(path: Path, value) -> None:
    """Write JSON in the destination directory and atomically replace it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(value, handle, indent=4, default=json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _default_dataset_loader(cfg):
    from dataset import load_dataset

    return load_dataset(cfg)


def _default_model_factory(cfg):
    from model import get_model

    return get_model(cfg)


def _build_prompt(mode, sample, cfg, index_sampler):
    if mode == "subq":
        return get_subq_prompt(sample, cfg)
    if mode == "suba":
        return get_suba_prompt(sample, cfg)
    if mode == "base":
        return get_base_prompt(sample, cfg)
    if mode == "refined":
        return get_refined_prompt(sample, cfg, index_sampler)
    raise ValueError(f"unsupported inference stage: {mode}")


def _prepare_manifest(cfg, qids, output_dir: Path) -> Path:
    manifest_path = output_dir / MANIFEST_FILENAME
    if manifest_path.is_file():
        manifest = load_manifest(manifest_path)
    else:
        manifest = create_manifest(cfg, qids)
        write_manifest(manifest_path, manifest)
    validate_manifest(manifest, cfg, qids)
    return manifest_path


def _refined_records(samples, split, generation_id):
    records = []
    for sample in samples:
        record = {
            "qid": sample["qid"],
            "split": split,
            "main_q": sample["main_q"],
            "gt_ans": sample["gt_ans"],
            "question_type": sample["question_type"],
            "base_answer": sample["base_answer"],
            "conf_base": sample["conf_base"],
            "refined_answer_list": sample["refined_answer_list"],
            "conf_refined": sample["conf_refined"],
            "generation_id": generation_id,
        }
        records.append(record)
    return records


def _validate_formatted_outputs(formatted, sample_qids, prompt_qids, mode, n):
    if not isinstance(formatted, Mapping):
        raise ValueError("formatted schema must be an object keyed by qid")

    expected_qids = set(sample_qids)
    actual_qids = set(formatted)
    if actual_qids != expected_qids:
        raise ValueError(
            "formatted qid mismatch: "
            f"missing={sorted(expected_qids - actual_qids)}, "
            f"unexpected={sorted(actual_qids - expected_qids)}"
        )

    value_key, confidence_key = OUTPUT_KEYS[mode]
    prompt_counts = Counter(prompt_qids)
    for qid in sample_qids:
        entry = formatted[qid]
        if not isinstance(entry, Mapping) or set(entry) != {value_key, confidence_key}:
            actual_keys = (
                sorted(entry) if isinstance(entry, Mapping) else type(entry).__name__
            )
            raise ValueError(
                f"formatted schema mismatch for qid {qid}: "
                f"expected {[value_key, confidence_key]}, got {actual_keys}"
            )
        confidence = entry[confidence_key]
        if not isinstance(confidence, Mapping) or set(confidence) != CONFIDENCE_KEYS:
            actual_keys = (
                sorted(confidence)
                if isinstance(confidence, Mapping)
                else type(confidence).__name__
            )
            raise ValueError(
                f"formatted schema mismatch for qid {qid}: expected confidence "
                f"keys {sorted(CONFIDENCE_KEYS)}, got {actual_keys}"
            )

        expected_count = prompt_counts[qid]
        if mode in ("suba", "refined"):
            values = entry[value_key]
            actual_count = len(values) if isinstance(values, list) else 1
            if not isinstance(values, list) or actual_count != expected_count:
                raise ValueError(
                    f"formatted count mismatch for qid {qid}: expected "
                    f"{expected_count}, got {actual_count} for {value_key}"
                )
            for confidence_name, confidence_values in confidence.items():
                actual_count = (
                    len(confidence_values)
                    if isinstance(confidence_values, list)
                    else 1
                )
                if (
                    not isinstance(confidence_values, list)
                    or actual_count != expected_count
                ):
                    raise ValueError(
                        f"formatted count mismatch for qid {qid}: expected "
                        f"{expected_count}, got {actual_count} for "
                        f"{confidence_key}.{confidence_name}"
                    )
        else:
            if expected_count != 1:
                raise ValueError(
                    f"formatted count mismatch for qid {qid}: expected one "
                    f"{mode} prompt, got {expected_count}"
                )
            if mode == "subq":
                values = entry[value_key]
                actual_count = len(values) if isinstance(values, list) else 1
                if not isinstance(values, list) or actual_count != int(n):
                    raise ValueError(
                        f"formatted count mismatch for qid {qid}: expected "
                        f"{int(n)}, got {actual_count} for {value_key}"
                    )
            elif not isinstance(entry[value_key], str):
                raise ValueError(
                    f"formatted schema mismatch for qid {qid}: "
                    f"{value_key} must be a string"
                )
            if any(isinstance(value, list) for value in confidence.values()):
                raise ValueError(
                    f"formatted schema mismatch for qid {qid}: "
                    f"{confidence_key} values must be scalars"
                )


def _validate_prompt_counts(cfg, sample_qids, prompt_qids, mode):
    if not sample_qids:
        raise ValueError(f"{mode} dataset has no samples")

    expected_count = {
        "subq": 1,
        "base": 1,
        "suba": int(cfg.runner_cfg.N),
        "refined": int(cfg.runner_cfg.K),
    }[mode]
    prompt_counts = Counter(prompt_qids)
    for qid in sample_qids:
        actual_count = prompt_counts[qid]
        if actual_count != expected_count:
            raise ValueError(
                f"prompt count mismatch for {mode} qid {qid}: "
                f"expected {expected_count}, got {actual_count}"
            )


def run_stage(
    cfg,
    model,
    *,
    dataset_loader: Optional[Callable] = None,
    prompt_builder: Optional[Callable] = None,
    output_formatter: Optional[Callable] = None,
):
    """Run one stage through the common dataset/prompt/generate/save boundary."""
    mode = str(cfg.runner_cfg.mode)
    if mode not in STAGE_ORDER:
        raise ValueError(f"unsupported inference stage: {mode}")

    dataset_loader = dataset_loader or _default_dataset_loader
    prompt_builder = prompt_builder or _build_prompt
    output_formatter = output_formatter or format_vllm_outputs
    logger = get_logger()
    output_dir = get_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = dataset_loader(cfg)
    index_sampler = None
    if mode == "refined":
        index_sampler = IndexSampler(
            cfg.runner_cfg.N,
            cfg.runner_cfg.M,
            cfg.runner_cfg.K,
        )

    samples = OrderedDict()
    prompts = []
    qids = []
    for original_sample in dataset:
        sample = dict(original_sample)
        qid = str(sample["qid"])
        sample["qid"] = qid
        samples[qid] = sample
        vision = sample.get("vision")
        text_prompt = prompt_builder(mode, sample, cfg, index_sampler)
        text_prompts = text_prompt if isinstance(text_prompt, list) else [text_prompt]
        if not all(isinstance(item, str) for item in text_prompts):
            raise ValueError(f"invalid text prompt for qid {qid}: {text_prompt!r}")
        for item in text_prompts:
            prompts.append(
                model.apply_chat_template(item, vision=vision, mm_uuids=qid)
            )
            qids.append(qid)

    sample_qids = list(samples)
    _validate_prompt_counts(cfg, sample_qids, qids, mode)
    manifest_path = _prepare_manifest(cfg, sample_qids, output_dir)
    started_manifest = mark_stage_started(manifest_path, mode)
    generation_id = started_manifest["stages"][mode]["generation_id"]
    logger.info("Stage %s: prepared %d prompts", mode, len(prompts))
    if prompts:
        logger.info("Stage %s first prompt: %s", mode, data_print(prompts[0]))
    outputs = model.generate(prompts)
    if len(outputs) != len(prompts):
        raise ValueError(
            f"generated output count mismatch: expected {len(prompts)}, "
            f"got {len(outputs)}"
        )
    formatted = output_formatter(mode, outputs, qids, cfg.runner_cfg.N)
    _validate_formatted_outputs(
        formatted,
        sample_qids,
        qids,
        mode,
        cfg.runner_cfg.N,
    )

    for qid, output_data in formatted.items():
        if qid in samples:
            samples[qid].update(output_data)

    output_path = output_dir / get_output_filename(cfg)
    write_json_atomic(output_path, formatted)
    if mode == "refined":
        records = _refined_records(
            samples.values(),
            str(cfg.dataset_cfg.split),
            generation_id,
        )
        write_json_atomic(output_dir / "refined_samples.json", records)
    mark_stage_complete(manifest_path, mode, generation_id)
    logger.info("Saved %s outputs to %s", mode, output_path)
    return formatted


def run_multi_stage(
    cfg,
    *,
    model_factory: Optional[Callable] = None,
    stage_runner: Optional[Callable] = None,
    **stage_kwargs,
):
    """Create one model and reuse it across all stages in dependency order."""
    model_factory = model_factory or _default_model_factory
    stage_runner = stage_runner or run_stage
    model = model_factory(cfg)
    results = OrderedDict()
    for mode in STAGE_ORDER:
        stage_cfg = cfg.for_stage(mode)
        model.cfg = stage_cfg
        results[mode] = stage_runner(stage_cfg, model, **stage_kwargs)
    return results


def run(
    cfg,
    *,
    model_factory: Optional[Callable] = None,
    stage_runner: Optional[Callable] = None,
    **stage_kwargs,
):
    """Run either the full pipeline or one legacy inference stage."""
    mode = str(cfg.runner_cfg.mode)
    if mode == "multi_stage":
        return run_multi_stage(
            cfg,
            model_factory=model_factory,
            stage_runner=stage_runner,
            **stage_kwargs,
        )
    if mode not in STAGE_ORDER:
        raise ValueError(
            f"unsupported runner mode {mode!r}; expected multi_stage or one of "
            f"{', '.join(STAGE_ORDER)}"
        )
    model_factory = model_factory or _default_model_factory
    stage_runner = stage_runner or run_stage
    stage_cfg = cfg.for_stage(mode)
    model = model_factory(cfg)
    model.cfg = stage_cfg
    return stage_runner(stage_cfg, model, **stage_kwargs)
