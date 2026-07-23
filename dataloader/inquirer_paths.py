from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


DEFAULT_SOURCE_ROOT = Path("data/inquirer-source")

_DATASET_CONFIG = {
    "dramaqa": (
        "data/dramaqa",
        "DRAMAQA_ROOT",
        "prompts",
        "data/dramaqa",
        "DRAMAQA_ROOT",
        "AnotherMissOh_images",
    ),
    "star": (
        "data/star",
        "STAR_DATASET_ROOT",
        "gen_starQA",
        "data/star/videos",
        "STAR_VIDEO_ROOT",
        None,
    ),
    "tvqa": (
        "data/tvqa",
        "TVQA_DATASET_ROOT",
        "gen_tvqa",
        "data/tvqa/videos",
        "TVQA_VIDEO_ROOT",
        None,
    ),
    "how2qa": (
        "data/how2qa",
        "HOW2QA_DATASET_ROOT",
        "gen_how2qakg",
        "data/how2qa/clips",
        "HOW2QA_VIDEO_ROOT",
        None,
    ),
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
        default_root, environment_variable, *_ = _DATASET_CONFIG[dataset_key]
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
    return resolve_dataset_root(
        dataset,
        dataset_root=dataset_root,
        environ=environ,
    ) / path


def resolve_media_root(
    dataset: str,
    *,
    media_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    dataset_key = dataset.lower()
    try:
        _, _, _, default_root, environment_variable, media_subdirectory = (
            _DATASET_CONFIG[dataset_key]
        )
    except KeyError as error:
        supported = ", ".join(sorted(_DATASET_CONFIG))
        raise ValueError(
            f"Unsupported INQUIRER dataset {dataset!r}; expected one of: {supported}"
        ) from error

    environment = os.environ if environ is None else environ
    configured_root = media_root or environment.get(environment_variable)
    root = Path(configured_root) if configured_root else Path(default_root)
    return root / media_subdirectory if media_subdirectory else root


def resolve_media_path(
    dataset: str,
    relative_path: str | Path,
    *,
    media_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    path = _safe_relative_path(relative_path)
    return resolve_media_root(
        dataset,
        media_root=media_root,
        environ=environ,
    ) / path


def resolve_inquirer_path(
    dataset: str,
    relative_path: str | Path,
    *,
    source_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    dataset_key = dataset.lower()
    try:
        _, _, source_directory, *_ = _DATASET_CONFIG[dataset_key]
    except KeyError as error:
        supported = ", ".join(sorted(_DATASET_CONFIG))
        raise ValueError(
            f"Unsupported INQUIRER dataset {dataset!r}; expected one of: {supported}"
        ) from error

    path = _safe_relative_path(relative_path)
    return (
        resolve_source_root(source_root, environ=environ)
        / source_directory
        / path
    )


def _safe_relative_path(relative_path: str | Path) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"INQUIRER input must be a safe relative path: {relative_path!s}"
        )
    return path
