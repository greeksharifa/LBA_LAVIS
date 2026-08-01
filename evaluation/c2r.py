"""Leakage-free dev-to-validation C2R evaluation."""

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path

import numpy as np

from dataset.mmmu_eval import evaluate_answer
from util.artifacts import STAGES, STAGE_DEPENDENCIES


TAU1_GRID = tuple(round(index / 10, 1) for index in range(11))
TAU2_GRID = tuple(round(-1 + index / 10, 1) for index in range(21))
BOOTSTRAP_SEED = 42
BOOTSTRAP_COUNT = 10_000
BOOTSTRAP_BATCH_SIZE = 512

_MANIFEST_NAME = "run_manifest.json"
_SAMPLES_NAME = "refined_samples.json"
_COMPATIBILITY_FIELDS = (
    "dataset",
    "model_name",
    "model_id",
    "N",
    "M",
    "K",
    "confidence_type",
    "num_data",
    "tensor_parallel_size",
    "enforce_eager",
    "swap_space",
    "limit_mm_per_prompt",
)
_REQUIRED_CONFIG_FIELDS = _COMPATIBILITY_FIELDS + (
    "split",
    "annotation_paths",
    "annotation_paths_resolved",
)
_REQUIRED_RECORD_FIELDS = (
    "qid",
    "split",
    "main_q",
    "gt_ans",
    "question_type",
    "base_answer",
    "conf_base",
    "refined_answer_list",
    "conf_refined",
    "generation_id",
)
_DATASET_ROLE_SPLITS = {
    "MMMU": {"dev": "val", "validation": "test"},
}


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number, got {value!r}")
    return value


def normalize_confidence(value, confidence_type):
    """Convert a configured confidence metric to higher-is-better [0, 1]."""
    value = _number(value, confidence_type)
    if confidence_type == "token_min_prob":
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"token_min_prob must be in [0, 1], got {value!r}"
            )
        return value
    if confidence_type == "seq_ppl":
        if value < 0.0:
            raise ValueError(f"seq_ppl must be non-negative, got {value!r}")
        return min(1.0, 1.0 / max(value, 1e-12))
    raise ValueError(f"unsupported confidence_type: {confidence_type!r}")


def select_refined_candidate(answers, confidences, confidence_type):
    """Select the first maximum-confidence refined candidate."""
    if not isinstance(answers, list) or not answers:
        raise ValueError("refined_answer_list must be a non-empty list")
    if not isinstance(confidences, list):
        raise ValueError("refined confidence values must be a list")
    if len(answers) != len(confidences):
        raise ValueError(
            "refined answer/confidence count mismatch: "
            f"answers={len(answers)}, confidences={len(confidences)}"
        )
    if not all(isinstance(answer, str) for answer in answers):
        raise ValueError("refined_answer_list values must be strings")
    normalized = [
        normalize_confidence(value, confidence_type) for value in confidences
    ]
    index = max(range(len(normalized)), key=normalized.__getitem__)
    return answers[index], normalized[index], index


def _metric_value(container, confidence_type, label, *, require_mapping=False):
    if isinstance(container, Mapping):
        if confidence_type not in container:
            raise ValueError(f"{label} missing confidence metric {confidence_type!r}")
        return container[confidence_type]
    if require_mapping:
        raise ValueError(f"{label} must be an object keyed by confidence metric")
    return container


def _validate_record(record, index):
    if not isinstance(record, Mapping):
        raise ValueError(f"record {index} must be an object")
    missing = [key for key in _REQUIRED_RECORD_FIELDS if key not in record]
    if missing:
        raise ValueError(f"record {index} missing required fields: {missing}")
    for key in ("qid", "split", "main_q", "question_type", "base_answer"):
        if not isinstance(record[key], str) or not record[key]:
            raise ValueError(f"record {index} field {key} must be a non-empty string")
    gold = record["gt_ans"]
    if not isinstance(gold, (str, list)):
        raise ValueError(f"record {index} field gt_ans must be a string or list")
    if isinstance(gold, list) and (
        not gold
        or not all(isinstance(answer, str) and answer for answer in gold)
    ):
        raise ValueError(
            f"record {index} field gt_ans list must contain non-empty strings"
        )
    if not isinstance(record["generation_id"], str) or not record["generation_id"]:
        raise ValueError(
            f"record {index} field generation_id must be a non-empty string"
        )


def _validated_candidates(record, index, confidence_type):
    base_raw = _metric_value(
        record["conf_base"], confidence_type, f"record {index} conf_base"
    )
    base_confidence = normalize_confidence(base_raw, confidence_type)
    refined_raw = _metric_value(
        record["conf_refined"],
        confidence_type,
        f"record {index} conf_refined",
        require_mapping=True,
    )
    refined = select_refined_candidate(
        record["refined_answer_list"], refined_raw, confidence_type
    )
    return base_confidence, refined


def prepare_samples(records, confidence_type, scorer=evaluate_answer):
    """Validate raw records and compute candidate choices/correctness once."""
    if not isinstance(records, list):
        raise ValueError("refined_samples.json must contain a list")
    if not callable(scorer):
        raise ValueError("scorer must be callable")
    prepared = []
    for index, record in enumerate(records):
        _validate_record(record, index)
        base_confidence, refined = _validated_candidates(
            record, index, confidence_type
        )
        refined_answer, refined_confidence, refined_index = refined
        score_args = (record["gt_ans"], record["question_type"])
        prepared.append(
            {
                "qid": record["qid"],
                "base_answer": record["base_answer"],
                "base_confidence": base_confidence,
                "base_correct": bool(scorer(record["base_answer"], *score_args)),
                "refined_answer": refined_answer,
                "refined_confidence": refined_confidence,
                "refined_index": refined_index,
                "refined_correct": bool(scorer(refined_answer, *score_args)),
            }
        )
    if not prepared:
        raise ValueError("refined_samples.json must contain at least one record")
    return prepared


def apply_thresholds(samples, tau1, tau2):
    """Apply one fixed gate without consulting ground truth for selection."""
    tau1 = _number(tau1, "tau1")
    tau2 = _number(tau2, "tau2")
    switched = []
    answers = []
    correct = []
    for sample in samples:
        refined_confidence = Decimal(str(sample["refined_confidence"]))
        refined_margin = Decimal(str(sample["base_confidence"])) + Decimal(
            str(tau2)
        )
        clears_refined_margin = refined_confidence >= refined_margin
        use_refined = (
            sample["base_confidence"] < tau1
            and clears_refined_margin
        )
        switched.append(use_refined)
        key = "refined" if use_refined else "base"
        answers.append(sample[f"{key}_answer"])
        correct.append(bool(sample[f"{key}_correct"]))
    return {
        "answers": answers,
        "correct": correct,
        "correct_count": sum(correct),
        "switch_count": sum(switched),
        "switched": switched,
    }


def search_thresholds(samples):
    """Search only the fixed dev grid with the specified deterministic ties."""
    best = None
    best_key = None
    for tau1 in TAU1_GRID:
        for tau2 in TAU2_GRID:
            result = apply_thresholds(samples, tau1, tau2)
            key = (
                result["correct_count"],
                -result["switch_count"],
                -tau1,
                tau2,
            )
            if best_key is None or key > best_key:
                best_key = key
                best = {"tau1": tau1, "tau2": tau2, **result}
    return best


def paired_bootstrap_delta(base_correct, gated_correct, *, seed=42, count=10_000):
    """Compute a deterministic paired percentile CI for gated-minus-base."""
    if isinstance(base_correct, (str, bytes)) or not isinstance(
        base_correct, Sequence
    ):
        raise ValueError("base_correct must be a sequence")
    if isinstance(gated_correct, (str, bytes)) or not isinstance(
        gated_correct, Sequence
    ):
        raise ValueError("gated_correct must be a sequence")
    if len(base_correct) != len(gated_correct) or not base_correct:
        raise ValueError("paired correctness vectors must have equal non-zero length")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap seed must be an integer")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("bootstrap count must be a positive integer")
    for label, values in (
        ("base_correct", base_correct),
        ("gated_correct", gated_correct),
    ):
        if any(
            isinstance(value, (str, bytes))
            or not isinstance(value, (bool, int, float, np.bool_, np.number))
            or value not in (0, 1)
            for value in values
        ):
            raise ValueError(f"{label} values must be binary boolean/0/1")
    differences = np.asarray(gated_correct, dtype=float) - np.asarray(
        base_correct, dtype=float
    )
    rng = np.random.default_rng(seed)
    bootstrap_deltas = np.empty(count, dtype=float)
    offset = 0
    while offset < count:
        batch_count = min(BOOTSTRAP_BATCH_SIZE, count - offset)
        sample_indices = rng.integers(
            0,
            len(differences),
            size=(batch_count, len(differences)),
        )
        bootstrap_deltas[offset : offset + batch_count] = differences[
            sample_indices
        ].mean(axis=1)
        offset += batch_count
    lower, upper = np.percentile(bootstrap_deltas, [2.5, 97.5])
    return {
        "delta": float(differences.mean()),
        "percentile_95_ci": [float(lower), float(upper)],
        "seed": seed,
        "count": count,
    }


def _load_json(path, label):
    if not path.is_file():
        raise FileNotFoundError(f"required {label} not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {label} {path}: {error}") from error


def _unique_qids(qids, label):
    if not isinstance(qids, list):
        raise ValueError(f"{label} qids must be a list")
    if not all(isinstance(qid, str) and qid for qid in qids):
        raise ValueError(f"{label} qids must be non-empty strings")
    if len(qids) != len(set(qids)):
        raise ValueError(f"{label} contains duplicate qid values")


def _validated_refined_stage(manifest, manifest_path):
    stages = manifest.get("stages")
    stage = stages.get("refined") if isinstance(stages, Mapping) else None
    if not isinstance(stage, Mapping) or not (
        stage.get("completed") is True and stage.get("state") == "completed"
    ):
        raise ValueError(
            f"refined stage is not completed in manifest {manifest_path}"
        )
    generation_id = stage.get("generation_id")
    if not isinstance(generation_id, str) or not generation_id:
        raise ValueError(
            f"refined stage generation_id missing in manifest {manifest_path}"
        )
    return stage


def _validate_completed_stage_lineage(manifest, manifest_path):
    stages = manifest.get("stages")
    if not isinstance(stages, Mapping):
        raise ValueError(f"manifest stages must be an object in {manifest_path}")
    if set(stages) != set(STAGES):
        raise ValueError(
            f"manifest stages must contain exactly {list(STAGES)} in {manifest_path}"
        )
    for stage_name in STAGES:
        status = stages[stage_name]
        if not isinstance(status, Mapping):
            raise ValueError(
                f"stage {stage_name} status must be an object in {manifest_path}"
            )
        if not (
            status.get("completed") is True
            and status.get("state") == "completed"
        ):
            continue
        generation_id = status.get("generation_id")
        if not isinstance(generation_id, str) or not generation_id:
            raise ValueError(
                f"stage {stage_name} generation_id missing in {manifest_path}"
            )
        parent_generations = status.get("parent_generations")
        dependencies = STAGE_DEPENDENCIES[stage_name]
        if not isinstance(parent_generations, Mapping) or set(
            parent_generations
        ) != set(dependencies):
            actual = (
                sorted(parent_generations)
                if isinstance(parent_generations, Mapping)
                else parent_generations
            )
            raise ValueError(
                f"stage {stage_name} parent_generations must contain exactly "
                f"{list(dependencies)}, got {actual!r}"
            )
        for parent_name in dependencies:
            expected_generation = parent_generations[parent_name]
            if not isinstance(expected_generation, str) or not expected_generation:
                raise ValueError(
                    f"stage {stage_name} parent_generations[{parent_name!r}] "
                    "must be a non-empty string"
                )
            parent_status = stages[parent_name]
            if not isinstance(parent_status, Mapping) or not (
                parent_status.get("completed") is True
                and parent_status.get("state") == "completed"
            ):
                raise ValueError(
                    f"stage {stage_name} parent_generations references incomplete "
                    f"parent {parent_name}"
                )
            active_generation = parent_status.get("generation_id")
            if active_generation != expected_generation:
                raise ValueError(
                    f"stage {stage_name} parent_generations mismatch for "
                    f"{parent_name}: recorded={expected_generation!r}, "
                    f"active={active_generation!r}"
                )


def _validate_runtime_config(config):
    tensor_parallel_size = config["tensor_parallel_size"]
    if (
        isinstance(tensor_parallel_size, bool)
        or not isinstance(tensor_parallel_size, int)
        or tensor_parallel_size <= 0
    ):
        raise ValueError(
            "run manifest tensor_parallel_size must be a positive integer"
        )
    if not isinstance(config["enforce_eager"], bool):
        raise ValueError("run manifest enforce_eager must be a boolean")
    swap_space = config["swap_space"]
    if (
        isinstance(swap_space, bool)
        or not isinstance(swap_space, float)
        or not math.isfinite(swap_space)
        or swap_space < 0.0
    ):
        raise ValueError(
            "run manifest swap_space must be a finite non-negative float"
        )
    limits = config["limit_mm_per_prompt"]
    if limits is None:
        return
    if not isinstance(limits, Mapping):
        raise ValueError(
            "run manifest limit_mm_per_prompt must be an object or null"
        )
    for modality, limit in limits.items():
        if not isinstance(modality, str) or not modality:
            raise ValueError(
                "run manifest limit_mm_per_prompt keys must be non-empty strings"
            )
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError(
                "run manifest limit_mm_per_prompt values must be "
                "non-negative integers"
            )


def _validated_annotation_paths(config):
    configured = config["annotation_paths"]
    resolved = config["annotation_paths_resolved"]
    if len(configured) != len(resolved):
        raise ValueError(
            "run manifest annotation path lists must have equal length"
        )
    if len(configured) != len(set(configured)) or len(resolved) != len(
        set(resolved)
    ):
        raise ValueError("run manifest annotation paths contain duplicates")
    paths = []
    for value in resolved:
        path = Path(value)
        if not path.is_absolute() or str(path.resolve()) != value:
            raise ValueError(
                "run manifest resolved annotation paths must be canonical absolute "
                f"paths, got {value!r}"
            )
        paths.append(path)
    return paths


def _selected_annotation_qids(config):
    dataset = config["dataset"]
    if dataset != "MMMU":
        raise ValueError(f"unsupported dataset for annotation provenance: {dataset!r}")
    all_qids = []
    for path in _validated_annotation_paths(config):
        annotations = _load_json(path, "annotation JSON")
        if not isinstance(annotations, list):
            raise ValueError(f"annotation JSON {path} must contain a list")
        for index, annotation in enumerate(annotations):
            if not isinstance(annotation, Mapping):
                raise ValueError(
                    f"annotation JSON {path} record {index} must be an object"
                )
            source_qid = annotation.get("question_id")
            if source_qid is None or not str(source_qid):
                raise ValueError(
                    f"annotation JSON {path} record {index} question_id must be "
                    "non-empty"
                )
            all_qids.append(str(source_qid))
    if not all_qids:
        raise ValueError("annotation provenance must contain at least one qid")
    _unique_qids(all_qids, "annotation provenance")
    num_data = config["num_data"]
    if num_data == -1 or num_data >= len(all_qids):
        return all_qids
    indices = np.linspace(0, len(all_qids) - 1, num_data, dtype=int)
    return [all_qids[index] for index in indices]


def _validate_manifest_integer_fields(config):
    for field in ("N", "M", "K"):
        value = config[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"run manifest {field} must be a positive integer")
    num_data = config["num_data"]
    if (
        isinstance(num_data, bool)
        or not isinstance(num_data, int)
        or (num_data != -1 and num_data <= 0)
    ):
        raise ValueError("run manifest num_data must be -1 or a positive integer")


def load_run(run_directory):
    """Load one fresh completed run and validate its exact provenance."""
    run_directory = Path(run_directory).resolve()
    manifest_path = run_directory / _MANIFEST_NAME
    samples_path = run_directory / _SAMPLES_NAME
    manifest = _load_json(manifest_path, "run manifest")
    records = _load_json(samples_path, "refined samples")
    if not isinstance(manifest, Mapping):
        raise ValueError("run manifest must be an object")
    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("run manifest config must be an object")
    missing = [field for field in _REQUIRED_CONFIG_FIELDS if field not in config]
    if missing:
        raise ValueError(
            "run manifest lacks fresh evaluation provenance fields: "
            f"{missing}"
        )
    _validate_manifest_integer_fields(config)
    _validate_runtime_config(config)
    split = config["split"]
    if not isinstance(split, str) or not split:
        raise ValueError("run manifest split must be explicit and non-empty")
    for key in ("annotation_paths", "annotation_paths_resolved"):
        paths = config[key]
        if not isinstance(paths, list) or not paths or not all(
            isinstance(path, str) and path for path in paths
        ):
            raise ValueError(f"run manifest {key} must be a non-empty string list")

    _validate_completed_stage_lineage(manifest, manifest_path)
    stage = _validated_refined_stage(manifest, manifest_path)
    manifest_qids = manifest.get("qids")
    _unique_qids(manifest_qids, "manifest")
    selected_annotation_qids = _selected_annotation_qids(config)
    if manifest_qids != selected_annotation_qids:
        raise ValueError(
            "annotation selection mismatch: manifest qids must exactly match "
            "the configured annotation selection in order"
        )
    if not isinstance(records, list):
        raise ValueError("refined_samples.json must contain a list")
    record_qids = []
    for index, record in enumerate(records):
        _validate_record(record, index)
        if record["split"] != split:
            raise ValueError(
                f"record split mismatch for qid {record['qid']}: "
                f"record={record['split']!r}, manifest={split!r}"
            )
        _validated_candidates(record, index, config["confidence_type"])
        if record["generation_id"] != stage["generation_id"]:
            raise ValueError(
                f"record generation mismatch for qid {record['qid']}: "
                f"record={record['generation_id']!r}, "
                f"manifest={stage['generation_id']!r}"
            )
        record_qids.append(record["qid"])
    _unique_qids(record_qids, "refined samples")
    if record_qids != manifest_qids:
        raise ValueError(
            "record qid mismatch: refined_samples.json qids must exactly match "
            "manifest qids in order"
        )
    final_manifest = _load_json(manifest_path, "run manifest")
    if not isinstance(final_manifest, Mapping):
        raise ValueError("run manifest must be an object")
    _validate_completed_stage_lineage(final_manifest, manifest_path)
    final_stage = _validated_refined_stage(final_manifest, manifest_path)
    stage_identity = (
        stage.get("completed"),
        stage.get("state"),
        stage.get("generation_id"),
        dict(stage.get("parent_generations", {})),
    )
    final_stage_identity = (
        final_stage.get("completed"),
        final_stage.get("state"),
        final_stage.get("generation_id"),
        dict(final_stage.get("parent_generations", {})),
    )
    if stage_identity != final_stage_identity:
        raise ValueError(
            f"refined stage changed while reading run artifacts at {run_directory}"
        )
    return {
        "run_directory": str(run_directory),
        "manifest_path": str(manifest_path),
        "samples_path": str(samples_path),
        "manifest": manifest,
        "config": dict(config),
        "records": records,
        "qids": manifest_qids,
    }


def _validate_run_pair(dev_run, validation_run):
    dev_config = dev_run["config"]
    validation_config = validation_run["config"]
    mismatches = [
        field
        for field in _COMPATIBILITY_FIELDS
        if dev_config[field] != validation_config[field]
    ]
    if mismatches:
        details = ", ".join(
            f"{field}: dev={dev_config[field]!r}, "
            f"validation={validation_config[field]!r}"
            for field in mismatches
        )
        raise ValueError(f"incompatible run manifests: {details}")
    dataset = dev_config["dataset"]
    role_splits = _DATASET_ROLE_SPLITS.get(dataset)
    if role_splits is None:
        raise ValueError(
            f"unsupported dataset for explicit evaluation roles: {dataset!r}"
        )
    if dev_config["split"] != role_splits["dev"]:
        raise ValueError(
            f"{dataset}: expected dev split {role_splits['dev']!r}, "
            f"got {dev_config['split']!r}"
        )
    if validation_config["split"] != role_splits["validation"]:
        raise ValueError(
            f"{dataset}: expected validation split "
            f"{role_splits['validation']!r}, got {validation_config['split']!r}"
        )
    qid_overlap = sorted(set(dev_run["qids"]) & set(validation_run["qids"]))
    if qid_overlap:
        raise ValueError(f"dev/validation qid overlap: {qid_overlap}")
    dev_paths = {
        str(Path(path).resolve())
        for path in dev_config["annotation_paths_resolved"]
    }
    validation_paths = {
        str(Path(path).resolve())
        for path in validation_config["annotation_paths_resolved"]
    }
    annotation_overlap = sorted(dev_paths & validation_paths)
    if annotation_overlap:
        raise ValueError(
            f"dev/validation annotation path overlap: {annotation_overlap}"
        )


def _accuracy(correct):
    return sum(correct) / len(correct)


def _split_report(
    run,
    threshold_run,
    samples,
    gated,
    tau1,
    tau2,
    seed,
    count,
):
    base_correct = [sample["base_correct"] for sample in samples]
    refined_correct = [sample["refined_correct"] for sample in samples]
    gated_correct = gated["correct"]
    base_accuracy = _accuracy(base_correct)
    gated_accuracy = _accuracy(gated_correct)
    absolute_delta = gated_accuracy - base_accuracy
    bootstrap = paired_bootstrap_delta(
        base_correct, gated_correct, seed=seed, count=count
    )
    config = run["config"]
    return {
        "source_split": config["split"],
        "annotation_paths": {
            "configured": list(config["annotation_paths"]),
            "resolved": list(config["annotation_paths_resolved"]),
        },
        "num_data": config["num_data"],
        "sample_count": len(samples),
        "qids": list(run["qids"]),
        "provenance": {
            "run_directory": run["run_directory"],
            "manifest": run["manifest_path"],
            "refined_samples": run["samples_path"],
            "manifest_config": config,
            "manifest_qids": list(run["qids"]),
            "refined_stage": dict(run["manifest"]["stages"]["refined"]),
        },
        "base_accuracy": base_accuracy,
        "raw_refined_accuracy": _accuracy(refined_correct),
        "gated_accuracy": gated_accuracy,
        "absolute_delta": absolute_delta,
        "relative_delta": (
            absolute_delta / base_accuracy if base_accuracy != 0 else None
        ),
        "paired_delta": bootstrap["delta"],
        "tau1": tau1,
        "tau2": tau2,
        "threshold_source": {
            "split": threshold_run["config"]["split"],
            "run_directory": threshold_run["run_directory"],
            "method": "fixed_dev_grid_search",
        },
        "switch_count": gated["switch_count"],
        "paired_95_ci": bootstrap["percentile_95_ci"],
        "bootstrap_seed": seed,
        "bootstrap_count": count,
    }


def evaluate_run_pair(
    dev_run_directory,
    validation_run_directory,
    *,
    scorer=evaluate_answer,
    bootstrap_seed=BOOTSTRAP_SEED,
    bootstrap_count=BOOTSTRAP_COUNT,
):
    """Select thresholds on dev once, then apply them unchanged to validation."""
    dev_run = load_run(dev_run_directory)
    validation_run = load_run(validation_run_directory)
    _validate_run_pair(dev_run, validation_run)
    confidence_type = dev_run["config"]["confidence_type"]
    dev_samples = prepare_samples(dev_run["records"], confidence_type, scorer)
    validation_samples = prepare_samples(
        validation_run["records"], confidence_type, scorer
    )
    selected = search_thresholds(dev_samples)
    tau1 = selected["tau1"]
    tau2 = selected["tau2"]
    validation_gated = apply_thresholds(validation_samples, tau1, tau2)
    return {
        "threshold_source": {
            "split": dev_run["config"]["split"],
            "run_directory": dev_run["run_directory"],
            "method": "fixed_dev_grid_search",
        },
        "confidence_type": confidence_type,
        "tau1": tau1,
        "tau2": tau2,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_count": bootstrap_count,
        "dev": _split_report(
            dev_run,
            dev_run,
            dev_samples,
            selected,
            tau1,
            tau2,
            bootstrap_seed,
            bootstrap_count,
        ),
        "validation": _split_report(
            validation_run,
            dev_run,
            validation_samples,
            validation_gated,
            tau1,
            tau2,
            bootstrap_seed,
            bootstrap_count,
        ),
    }


def write_report_atomic(path, report):
    """Write a report by atomically replacing it from the same directory."""
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
            json.dump(report, handle, indent=4)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
