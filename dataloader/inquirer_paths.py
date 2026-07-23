from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


DEFAULT_SOURCE_ROOT = Path("data/inquirer-source")

_DATASET_CONFIG = {
    "dramaqa": ("data/dramaqa", "DRAMAQA_ROOT", "prompts"),
    "star": ("data/star", "STAR_DATASET_ROOT", "gen_starQA"),
    "tvqa": ("data/tvqa", "TVQA_DATASET_ROOT", "gen_tvqa"),
    "how2qa": ("data/how2qa", "HOW2QA_DATASET_ROOT", "gen_how2qakg"),
}


def resolve_source_root(
    source_root: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    environment = os.environ if environ is None else environ
    configured_root = source_root or environment.get("INQUIRER_SOURCE_ROOT")
    return Path(configured_root) if configured_root else DEFAULT_SOURCE_ROOT


def resolve_dataset_root(
    dataset: str,
    *,
    dataset_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    dataset_key = dataset.lower()
    try:
        default_root, environment_variable, _ = _DATASET_CONFIG[dataset_key]
    except KeyError as error:
        supported = ", ".join(sorted(_DATASET_CONFIG))
        raise ValueError(
            f"Unsupported INQUIRER dataset {dataset!r}; expected one of: {supported}"
        ) from error

    environment = os.environ if environ is None else environ
    configured_root = dataset_root or environment.get(environment_variable)
    return Path(configured_root) if configured_root else Path(default_root)


def resolve_dataset_path(
    dataset: str,
    relative_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    path = _safe_relative_path(relative_path)
    return _resolve_contained(
        resolve_dataset_root(
            dataset,
            dataset_root=dataset_root,
            environ=environ,
        ),
        path,
    )


def resolve_inquirer_path(
    dataset: str,
    relative_path: str | Path,
    *,
    source_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    dataset_key = dataset.lower()
    try:
        _, _, source_directory = _DATASET_CONFIG[dataset_key]
    except KeyError as error:
        supported = ", ".join(sorted(_DATASET_CONFIG))
        raise ValueError(
            f"Unsupported INQUIRER dataset {dataset!r}; expected one of: {supported}"
        ) from error

    path = _safe_relative_path(relative_path)
    return _resolve_contained(
        resolve_source_root(source_root, environ=environ),
        Path(source_directory) / path,
    )


def _resolve_contained(root: str | Path, relative_path: Path) -> Path:
    resolved_root = Path(root).expanduser().resolve(strict=False)
    candidate = (resolved_root / relative_path).resolve(strict=False)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(
            f"INQUIRER path escapes configured root: {relative_path!s}"
        ) from error
    return candidate


def _safe_relative_path(relative_path: str | Path) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"INQUIRER input must be a safe relative path: {relative_path!s}"
        )
    return path
