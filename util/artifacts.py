import fcntl
import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


MANIFEST_FILENAME = "run_manifest.json"
STAGES = ("subq", "suba", "base", "refined")


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
    }


def create_manifest(cfg, qids: Iterable[Any]) -> Dict[str, Any]:
    """Create a new run manifest for the final selected dataset qids."""
    return {
        "config": core_run_config(cfg),
        "qids": [str(qid) for qid in qids],
        "stages": {stage: {"completed": False} for stage in STAGES},
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


def mark_stage_started(
    path: Path,
    stage: str,
    generation_id: str = None,
) -> Dict[str, Any]:
    """Atomically mark a stage as the active, incomplete generation."""
    if stage not in STAGES:
        raise ValueError(f"unknown manifest stage: {stage}")
    path = Path(path)
    with _manifest_lock(path):
        manifest = load_manifest(path)
        stage_status = manifest.setdefault("stages", {}).setdefault(stage, {})
        stage_status.update(
            {
                "completed": False,
                "state": "running",
                "generation_id": generation_id or uuid.uuid4().hex,
            }
        )
        write_manifest(path, manifest)
    return manifest


def mark_stage_complete(
    path: Path,
    stage: str,
    generation_id: str = None,
) -> Dict[str, Any]:
    """Mark one stage complete if it is still the requested generation."""
    if stage not in STAGES:
        raise ValueError(f"unknown manifest stage: {stage}")
    path = Path(path)
    with _manifest_lock(path):
        manifest = load_manifest(path)
        stage_status = manifest.setdefault("stages", {}).setdefault(stage, {})
        active_generation_id = stage_status.get("generation_id")
        if generation_id is not None and active_generation_id != generation_id:
            raise ValueError(
                f"stage {stage} generation changed: active={active_generation_id!r}, "
                f"completed={generation_id!r}"
            )
        stage_status.update({"completed": True, "state": "completed"})
        if generation_id is not None:
            stage_status["generation_id"] = generation_id
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
    return manifest
