from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "dataloader" / "inquirer_paths.py"


def load_path_module():
    assert MODULE_PATH.is_file(), "dataloader/inquirer_paths.py must exist"
    spec = importlib.util.spec_from_file_location("inquirer_paths", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_split_module():
    module_path = REPOSITORY_ROOT / "split_data.py"
    spec = importlib.util.spec_from_file_location("split_data", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    package = types.ModuleType("dataloader")
    package.__path__ = [str(REPOSITORY_ROOT / "dataloader")]
    saved_package = sys.modules.get("dataloader")
    saved_paths = sys.modules.get("dataloader.inquirer_paths")
    sys.modules["dataloader"] = package
    sys.modules["dataloader.inquirer_paths"] = load_path_module()
    try:
        spec.loader.exec_module(module)
    finally:
        if saved_package is None:
            sys.modules.pop("dataloader", None)
        else:
            sys.modules["dataloader"] = saved_package
        if saved_paths is None:
            sys.modules.pop("dataloader.inquirer_paths", None)
        else:
            sys.modules["dataloader.inquirer_paths"] = saved_paths
    return module


@pytest.mark.parametrize(
    ("dataset", "source_directory"),
    [
        ("dramaqa", "prompts"),
        ("star", "gen_starQA"),
        ("tvqa", "gen_tvqa"),
        ("how2qa", "gen_how2qakg"),
    ],
)
def test_resolves_generated_qa_inputs_below_source_root(
    tmp_path: Path,
    dataset: str,
    source_directory: str,
) -> None:
    paths = load_path_module()
    source_root = tmp_path / "inquirer-source"
    expected = source_root / source_directory / "generated_qas.json"
    expected.parent.mkdir(parents=True)
    expected.touch()

    actual = paths.resolve_inquirer_path(
        dataset,
        "generated_qas.json",
        source_root=source_root,
        environ={},
    )

    assert actual == expected


@pytest.mark.parametrize(
    ("dataset", "dataset_environment"),
    [
        ("star", "STAR_DATASET_ROOT"),
        ("tvqa", "TVQA_DATASET_ROOT"),
        ("how2qa", "HOW2QA_DATASET_ROOT"),
    ],
)
def test_dataset_assets_use_dedicated_environment_roots(
    tmp_path: Path,
    dataset: str,
    dataset_environment: str,
) -> None:
    paths = load_path_module()
    dataset_root = tmp_path / "metadata-and-features"

    asset_path = paths.resolve_dataset_path(
        dataset,
        "features.pth",
        environ={dataset_environment: str(dataset_root)},
    )

    assert asset_path == (dataset_root / "features.pth").resolve()


def test_dramaqa_assets_use_dramaqa_root(
    tmp_path: Path,
) -> None:
    paths = load_path_module()
    dataset_root = tmp_path / "dramaqa"

    asset_path = paths.resolve_dataset_path(
        "dramaqa",
        "clipvitl14.pth",
        environ={"DRAMAQA_ROOT": str(dataset_root)},
    )

    assert asset_path == (dataset_root / "clipvitl14.pth").resolve()


@pytest.mark.parametrize(
    ("dataset", "dataset_root"),
    [
        ("star", "data/star"),
        ("tvqa", "data/tvqa"),
        ("how2qa", "data/how2qa"),
    ],
)
def test_dataset_roots_have_neutral_defaults(
    dataset: str,
    dataset_root: str,
) -> None:
    paths = load_path_module()

    assert paths.resolve_dataset_root(dataset, environ={}) == Path(dataset_root)


def test_resolver_rejects_internal_symlink_escape(tmp_path: Path) -> None:
    paths = load_path_module()
    dataset_root = tmp_path / "dataset"
    outside_root = tmp_path / "outside"
    dataset_root.mkdir()
    outside_root.mkdir()
    (dataset_root / "escape").symlink_to(outside_root, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes configured root"):
        paths.resolve_dataset_path(
            "star",
            "escape/features.pth",
            dataset_root=dataset_root,
            environ={},
        )


def test_resolver_allows_root_itself_to_be_a_symlink(tmp_path: Path) -> None:
    paths = load_path_module()
    real_root = tmp_path / "shared-dataset"
    configured_root = tmp_path / "dataset-link"
    real_root.mkdir()
    configured_root.symlink_to(real_root, target_is_directory=True)

    actual = paths.resolve_dataset_path(
        "star",
        "features.pth",
        dataset_root=configured_root,
        environ={},
    )

    assert actual == (real_root / "features.pth").resolve()


def test_source_root_uses_environment_then_neutral_default(tmp_path: Path) -> None:
    paths = load_path_module()

    assert paths.resolve_source_root(
        environ={"INQUIRER_SOURCE_ROOT": str(tmp_path)}
    ) == tmp_path
    assert paths.resolve_source_root(environ={}) == Path("data/inquirer-source")


@pytest.mark.parametrize(
    "unsafe_path",
    ["../outside.json", "nested/../../outside.json", "/absolute/input.json"],
)
def test_rejects_paths_outside_configured_dataset_root(unsafe_path: str) -> None:
    paths = load_path_module()

    with pytest.raises(ValueError, match="relative path"):
        paths.resolve_inquirer_path(
            "dramaqa",
            unsafe_path,
            source_root=Path("source"),
            environ={},
        )


def test_rejects_unknown_dataset() -> None:
    paths = load_path_module()

    with pytest.raises(ValueError, match="Unsupported INQUIRER dataset"):
        paths.resolve_dataset_path(
            "unknown",
            "generated_qas.json",
            environ={},
        )


def test_path_module_has_no_heavy_imports_or_fixed_server_paths() -> None:
    source = MODULE_PATH.read_text()

    assert "torch" not in source
    assert "/" + "home/" not in source
    assert "/data" not in source
    assert "/" + "nas-" not in source


@pytest.mark.parametrize(
    "relative_path",
    [
        "dataloader/dramaqa.py",
        "dataloader/dramaqa_kg.py",
        "dataloader/star.py",
        "dataloader/tvqa.py",
        "dataloader/how2qa.py",
        "split_data.py",
    ],
)
def test_augmented_data_code_uses_resolver_without_fixed_roots(
    relative_path: str,
) -> None:
    source = (REPOSITORY_ROOT / relative_path).read_text()

    assert "resolve_dataset_path" in source
    assert "/" + "home/" not in source
    assert "/data" not in source
    assert "/" + "nas-" not in source
    assert "pdb." + "set_trace" not in source
    assert "break" + "point(" not in source


@pytest.mark.parametrize(
    ("relative_path", "argument_name"),
    [
        ("dataloader/dramaqa.py", "dramaqa_root"),
        ("dataloader/dramaqa_kg.py", "dramaqa_root"),
        ("dataloader/star.py", "star_dataset_root"),
        ("dataloader/tvqa.py", "tvqa_dataset_root"),
        ("dataloader/how2qa.py", "how2qa_dataset_root"),
    ],
)
def test_loaders_pass_dataset_root_arguments_to_resolver(
    relative_path: str,
    argument_name: str,
) -> None:
    source = (REPOSITORY_ROOT / relative_path).read_text()

    assert argument_name in source


@pytest.mark.parametrize("entrypoint", ["train.py", "eval.py"])
def test_entrypoints_expose_configurable_inquirer_source_root(
    entrypoint: str,
) -> None:
    source = (REPOSITORY_ROOT / entrypoint).read_text()

    assert "--inquirer-source-root" in source
    assert "INQUIRER_SOURCE_ROOT" in source
    assert "data/inquirer-source" in source
    for option in (
        "--dramaqa-root",
        "--star-dataset-root",
        "--tvqa-dataset-root",
        "--how2qa-dataset-root",
    ):
        assert option in source


def test_augmentation_branch_has_no_dead_media_root_surface() -> None:
    resolver_source = MODULE_PATH.read_text()
    entrypoint_source = (
        (REPOSITORY_ROOT / "train.py").read_text()
        + (REPOSITORY_ROOT / "eval.py").read_text()
    )
    readme_source = (REPOSITORY_ROOT / "README.md").read_text()

    assert "resolve_media_" not in resolver_source
    assert "--star-video-root" not in entrypoint_source
    assert "--tvqa-video-root" not in entrypoint_source
    assert "--how2qa-video-root" not in entrypoint_source
    assert "STAR_VIDEO_ROOT" not in readme_source
    assert "TVQA_VIDEO_ROOT" not in readme_source
    assert "HOW2QA_VIDEO_ROOT" not in readme_source


def test_readme_documents_precomputed_dataset_roots() -> None:
    source = (REPOSITORY_ROOT / "README.md").read_text()

    for environment_variable in (
        "STAR_DATASET_ROOT",
        "TVQA_DATASET_ROOT",
        "HOW2QA_DATASET_ROOT",
    ):
        assert environment_variable in source
    assert "precomputed features" in source


def test_build_dramaqa_split_reads_and_writes_configured_roots(
    tmp_path: Path,
) -> None:
    split_data = load_split_module()
    dataset_root = tmp_path / "dramaqa"
    source_root = tmp_path / "inquirer-source"
    dataset_root.mkdir()
    (source_root / "prompts").mkdir(parents=True)
    (dataset_root / "AnotherMissOhQA_train_set_ori_scsh.json").write_text(
        '[{"qid": 1}]'
    )
    (source_root / "prompts/AnotherMissOhQA_train_set_naive_sceneprob.json").write_text(
        '[{"qid": 2, "perplex": 0.1}, {"qid": 3, "perplex": 0.9}]'
    )
    (source_root / "prompts/AnotherMissOhQA_train_set_naive_shotprob.json").write_text(
        '[{"qid": 4, "perplex": 0.2}, {"qid": 5, "perplex": 0.8}]'
    )

    output_path = split_data.build_dramaqa_split(
        source_root=source_root,
        dataset_root=dataset_root,
        filter_ratio=0.5,
    )

    assert output_path == (
        dataset_root / "dramaqa_train_filtered_r0p5_Naive.json"
    ).resolve()
    assert split_data.json.loads(output_path.read_text()) == [
        {"qid": 1},
        {"qid": 2, "perplex": 0.1},
        {"qid": 4, "perplex": 0.2},
    ]


def test_split_output_names_encode_ratio_without_collisions(
    tmp_path: Path,
) -> None:
    split_data = load_split_module()
    dataset_root = tmp_path / "dramaqa"
    source_root = tmp_path / "inquirer-source"
    dataset_root.mkdir()
    (source_root / "prompts").mkdir(parents=True)
    (dataset_root / "AnotherMissOhQA_train_set_ori_scsh.json").write_text("[]")
    for name in ("scene", "shot"):
        (
            source_root
            / f"prompts/AnotherMissOhQA_train_set_naive_{name}prob.json"
        ).write_text('[{"perplex": 0.1}]')

    ratio_half = split_data.build_dramaqa_split(
        source_root=source_root,
        dataset_root=dataset_root,
        filter_ratio=0.5,
    )
    ratio_five_percent = split_data.build_dramaqa_split(
        source_root=source_root,
        dataset_root=dataset_root,
        filter_ratio=0.05,
    )

    assert ratio_half.name == "dramaqa_train_filtered_r0p5_Naive.json"
    assert ratio_five_percent.name == "dramaqa_train_filtered_r0p05_Naive.json"
    assert ratio_half != ratio_five_percent


def test_split_rejects_output_symlink_escape(tmp_path: Path) -> None:
    split_data = load_split_module()
    dataset_root = tmp_path / "dramaqa"
    source_root = tmp_path / "inquirer-source"
    outside_root = tmp_path / "outside"
    dataset_root.mkdir()
    outside_root.mkdir()
    (source_root / "prompts").mkdir(parents=True)
    (dataset_root / "AnotherMissOhQA_train_set_ori_scsh.json").write_text("[]")
    for name in ("scene", "shot"):
        (
            source_root
            / f"prompts/AnotherMissOhQA_train_set_naive_{name}prob.json"
        ).write_text("[]")
    output_path = dataset_root / "dramaqa_train_filtered_r0p5_Naive.json"
    output_path.symlink_to(outside_root / "escaped.json")

    with pytest.raises(ValueError, match="escapes configured root"):
        split_data.build_dramaqa_split(
            source_root=source_root,
            dataset_root=dataset_root,
            filter_ratio=0.5,
        )
