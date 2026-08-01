import fcntl
import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional


MANIFEST_FILENAME = "run_manifest.json"
STAGES = ("subq", "suba", "base", "refined")
STAGE_DEPENDENCIES = {
    "subq": (),
    "suba": ("subq",),
    "base": (),
    "refined": ("subq", "suba", "base"),
}
STAGE_DEPENDENTS = {
    "subq": ("suba", "refined"),
    "suba": ("refined",),
    "base": ("refined",),
    "refined": (),
}


def core_run_config(cfg) -> Dict[str, Any]:
    """Return the config fields that define artifact compatibility."""
    runner_cfg = cfg.runner_cfg
    dataset_cfg = cfg.dataset_cfg
    model_cfg = cfg.model_cfg
    split = str(dataset_cfg.split)
    configured_annotation_paths = [
        str(path) for path in dataset_cfg.ann_paths[split]
    ]
    root_dir = Path(str(dataset_cfg.root_dir))
    resolved_annotation_paths = [
        str((root_dir / path).resolve()) for path in configured_annotation_paths
    ]
    configured_num_data = dataset_cfg.get("num_data", -1)
    if isinstance(configured_num_data, Mapping):
        configured_num_data = configured_num_data.get(split, -1)
    limit_config = dataset_cfg.get("limit_mm_per_prompt", None)
    normalized_limit_config = None
    if limit_config is not None:
        normalized_limit_config = {
            str(name): int(limit) for name, limit in limit_config.items()
        }

    return {
        "dataset": str(dataset_cfg.dataset_name),
        "split": split,
        "model_name": str(model_cfg.model_name),
        "model_id": model_cfg.get("model_id", None),
        "N": int(runner_cfg.N),
        "M": int(runner_cfg.M),
        "K": int(runner_cfg.K),
        "confidence_type": str(runner_cfg.confidence_type),
        "annotation_paths": configured_annotation_paths,
        "annotation_paths_resolved": resolved_annotation_paths,
        "num_data": int(configured_num_data),
        "tensor_parallel_size": int(model_cfg.tensor_parallel_size),
        "enforce_eager": bool(model_cfg.enforce_eager),
        "swap_space": float(model_cfg.swap_space),
        "limit_mm_per_prompt": normalized_limit_config,
    }


def create_manifest(cfg, qids: Iterable[Any]) -> Dict[str, Any]:
    """Create a new run manifest for the final selected dataset qids."""
    return {
        "config": core_run_config(cfg),
        "qids": [str(qid) for qid in qids],
        "stages": {stage: {"completed": False} for stage in STAGES},
        "generation_history": {stage: [] for stage in STAGES},
    }


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Atomically write a manifest by replacing it from the same directory."""
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
            json.dump(manifest, handle, indent=4)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_manifest(path: Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@contextmanager
def _manifest_lock(path: Path):
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _require_generation_id(generation_id: str, label: str) -> str:
    if not isinstance(generation_id, str) or not generation_id:
        raise ValueError(f"{label} generation_id must be a non-empty string")
    return generation_id


def _validated_generation_history(manifest: Dict[str, Any]) -> Dict[str, list]:
    is_legacy = "generation_history" not in manifest
    if is_legacy:
        history = {stage: [] for stage in STAGES}
        manifest["generation_history"] = history
    else:
        history = manifest["generation_history"]
    if not isinstance(history, Mapping):
        raise ValueError("manifest generation_history must be an object")
    if set(history) != set(STAGES):
        raise ValueError(
            "manifest generation_history must contain exactly "
            f"{list(STAGES)}"
        )

    validated = {}
    for stage in STAGES:
        stage_history = history[stage]
        if not isinstance(stage_history, list):
            raise ValueError(
                f"manifest generation_history[{stage!r}] must be a list"
            )
        if not all(isinstance(value, str) and value for value in stage_history):
            raise ValueError(
                f"manifest generation_history[{stage!r}] must contain "
                "non-empty strings"
            )
        if len(stage_history) != len(set(stage_history)):
            raise ValueError(
                f"manifest generation_history[{stage!r}] contains duplicates"
            )
        validated[stage] = stage_history

    stages = manifest.get("stages")
    if not isinstance(stages, Mapping):
        raise ValueError("manifest stages must be an object")
    for stage in STAGES:
        status = stages.get(stage)
        if not isinstance(status, Mapping):
            continue
        visible_generation = status.get("generation_id")
        if visible_generation is None:
            continue
        _require_generation_id(visible_generation, f"stage {stage}")
        if visible_generation not in validated[stage]:
            if not is_legacy:
                raise ValueError(
                    "manifest generation_history is missing visible "
                    f"generation {visible_generation!r} for stage {stage}"
                )
            validated[stage].append(visible_generation)
    return validated


def _validated_parent_generations(
    manifest: Mapping[str, Any],
    stage: str,
    parent_generations: Optional[Mapping[str, str]],
) -> Dict[str, str]:
    dependencies = STAGE_DEPENDENCIES[stage]
    expected = set(dependencies)
    if parent_generations is None:
        if expected:
            raise ValueError(
                f"stage {stage} parent generations must contain exactly "
                f"{sorted(expected)}"
            )
        return {}
    if not isinstance(parent_generations, Mapping):
        raise ValueError(f"stage {stage} parent generations must be an object")
    actual = set(parent_generations)
    if actual != expected:
        raise ValueError(
            f"stage {stage} parent generations must contain exactly "
            f"{sorted(expected)}, got {sorted(actual)}"
        )

    stages = manifest.get("stages")
    if not isinstance(stages, Mapping):
        raise ValueError("manifest stages must be an object")
    validated = {}
    for parent in dependencies:
        expected_generation = parent_generations[parent]
        _require_generation_id(
            expected_generation,
            f"stage {stage} parent generations[{parent!r}]",
        )
        status = stages.get(parent)
        if not isinstance(status, Mapping) or not (
            status.get("completed") is True
            and status.get("state") == "completed"
        ):
            raise ValueError(f"parent generation unavailable for {parent}")
        active_generation = status.get("generation_id")
        _require_generation_id(active_generation, f"parent {parent}")
        if active_generation != expected_generation:
            raise ValueError(
                f"parent generation changed for {parent}: "
                f"active={active_generation!r}, expected={expected_generation!r}"
            )
        validated[parent] = expected_generation
    return validated


def mark_stage_started(
    path: Path,
    stage: str,
    generation_id: str = None,
    *,
    parent_generations: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Start one generation and atomically invalidate its dependents."""
    if stage not in STAGES:
        raise ValueError(f"unknown manifest stage: {stage}")
    if generation_id is None:
        generation_id = uuid.uuid4().hex
    else:
        _require_generation_id(generation_id, f"stage {stage}")
    path = Path(path)
    with _manifest_lock(path):
        manifest = load_manifest(path)
        generation_history = _validated_generation_history(manifest)
        validated_parents = _validated_parent_generations(
            manifest,
            stage,
            parent_generations,
        )
        stage_status = manifest.setdefault("stages", {}).setdefault(stage, {})
        if stage_status.get("generation_id") == generation_id:
            raise ValueError(
                f"stage {stage} generation_id {generation_id!r} is already active"
            )
        if generation_id in generation_history[stage]:
            raise ValueError(
                f"stage {stage} generation_id {generation_id!r} was previously used"
            )
        generation_history[stage].append(generation_id)
        stage_status.clear()
        stage_status.update(
            {
                "completed": False,
                "state": "running",
                "generation_id": generation_id,
                "parent_generations": validated_parents,
            }
        )
        for dependent in STAGE_DEPENDENTS[stage]:
            dependent_status = manifest["stages"].setdefault(dependent, {})
            dependent_status.clear()
            dependent_status.update(
                {
                    "completed": False,
                    "state": "invalidated",
                    "invalidated_by": {
                        "stage": stage,
                        "generation_id": generation_id,
                    },
                }
            )
        write_manifest(path, manifest)
    return manifest


def mark_stage_complete(
    path: Path,
    stage: str,
    generation_id: str = None,
    *,
    artifact_writer: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Guard artifact promotion and completion with generation lineage."""
    if stage not in STAGES:
        raise ValueError(f"unknown manifest stage: {stage}")
    _require_generation_id(generation_id, f"stage {stage}")
    if artifact_writer is not None and not callable(artifact_writer):
        raise ValueError("artifact_writer must be callable")
    path = Path(path)
    with _manifest_lock(path):
        manifest = load_manifest(path)
        stage_status = manifest.setdefault("stages", {}).setdefault(stage, {})
        active_generation_id = stage_status.get("generation_id")
        if active_generation_id != generation_id:
            raise ValueError(
                f"stage {stage} generation changed: active={active_generation_id!r}, "
                f"completed={generation_id!r}"
            )
        if (
            stage_status.get("completed") is not False
            or stage_status.get("state") != "running"
        ):
            raise ValueError(
                f"stage {stage} generation {generation_id!r} is not running"
            )
        _validated_parent_generations(
            manifest,
            stage,
            stage_status.get("parent_generations"),
        )
        if artifact_writer is not None:
            artifact_writer()
        stage_status.update({"completed": True, "state": "completed"})
        write_manifest(path, manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any], cfg, qids: Iterable[Any]) -> None:
    """Require the manifest's core config and exact qid set to match the run."""
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be an object")
    expected_config = core_run_config(cfg)
    actual_config = manifest.get("config")
    if not isinstance(actual_config, Mapping):
        raise ValueError("manifest config must be an object")
    mismatched_fields = [
        key
        for key, expected_value in expected_config.items()
        if actual_config.get(key) != expected_value
    ]
    if mismatched_fields:
        details = ", ".join(
            f"{key}={actual_config.get(key)!r} (expected {expected_config[key]!r})"
            for key in mismatched_fields
        )
        raise ValueError(f"manifest config mismatch: {details}")

    expected_qids = [str(qid) for qid in qids]
    manifest_qids = manifest.get("qids")
    if not isinstance(manifest_qids, list):
        raise ValueError("manifest qids must be a list")
    actual_qids = [str(qid) for qid in manifest_qids]
    if len(actual_qids) != len(expected_qids) or set(actual_qids) != set(expected_qids):
        missing = sorted(set(expected_qids) - set(actual_qids))
        unexpected = sorted(set(actual_qids) - set(expected_qids))
        raise ValueError(
            "manifest qid mismatch: "
            f"missing={missing}, unexpected={unexpected}, "
            f"expected_count={len(expected_qids)}, actual_count={len(actual_qids)}"
        )


def validate_completed_stage(
    path: Path,
    stage: str,
    cfg,
    qids: Iterable[Any],
) -> Dict[str, Any]:
    """Validate compatibility and completion before consuming a stage artifact."""
    if stage not in STAGES:
        raise ValueError(f"unknown manifest stage: {stage}")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"required {stage} manifest not found: {path}")
    try:
        manifest = load_manifest(path)
        validate_manifest(manifest, cfg, qids)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            f"required {stage} manifest invalid at {path}: {error}"
        ) from error

    stage_status = manifest.get("stages", {}).get(stage)
    if not isinstance(stage_status, Mapping):
        raise ValueError(f"required {stage} status missing in manifest {path}")
    if (
        stage_status.get("completed") is not True
        or stage_status.get("state") != "completed"
    ):
        raise ValueError(
            f"required {stage} stage is not completed in manifest {path}: "
            f"completed={stage_status.get('completed')!r}, "
            f"state={stage_status.get('state')!r}"
        )
    generation_id = stage_status.get("generation_id")
    if not isinstance(generation_id, str) or not generation_id:
        raise ValueError(
            f"required {stage} generation_id missing in manifest {path}"
        )
    return manifest
