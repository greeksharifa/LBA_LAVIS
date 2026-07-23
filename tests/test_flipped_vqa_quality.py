from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUGMENTATION_PATH = (
    REPOSITORY_ROOT / "dataloader" / "inquirer_augmentation.py"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_augmentation_module():
    assert AUGMENTATION_PATH.is_file()
    return load_module("inquirer_augmentation", AUGMENTATION_PATH)


def load_entrypoint_parser(path: Path):
    augmentation = load_augmentation_module()
    tree = ast.parse(path.read_text())
    parser_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_args_parser"
    )
    namespace = {
        "argparse": argparse,
        "os": os,
        "removal_ratio": augmentation.removal_ratio,
        "positive_int": augmentation.positive_int,
    }
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[parser_function], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace["get_args_parser"]()


@pytest.mark.parametrize("removal_ratio", [0.0, 0.25, 1.0])
def test_filter_generated_items_uses_stable_low_perplexity_order(
    removal_ratio: float,
) -> None:
    augmentation = load_augmentation_module()
    items = [
        {"id": "a", "perplex": 0.2},
        {"id": "b", "perplex": 0.1},
        {"id": "c", "perplex": 0.1},
        {"id": "d", "perplex": 0.3},
    ]
    expected = {
        0.0: ["b", "c", "a", "d"],
        0.25: ["b", "c", "a"],
        1.0: [],
    }[removal_ratio]

    selected = augmentation.filter_generated_items(items, removal_ratio)

    assert [item["id"] for item in selected] == expected


def test_local_sampling_is_reproducible_rank_independent_and_rng_safe() -> None:
    augmentation = load_augmentation_module()
    items = [{"id": index} for index in range(8)]
    random.seed(1234)
    global_state = random.getstate()

    rank_zero = augmentation.sample_generated_items(items, count=3, seed=7)
    rank_three = augmentation.sample_generated_items(items, count=3, seed=7)

    assert rank_zero == rank_three
    assert random.getstate() == global_state
    assert augmentation.sample_generated_items(items, count=3, seed=8) != rank_zero


@pytest.mark.parametrize("invalid_count", [0, -1, 1.5])
def test_local_sampling_rejects_invalid_counts(invalid_count) -> None:
    augmentation = load_augmentation_module()

    with pytest.raises((TypeError, ValueError), match="positive integer"):
        augmentation.sample_generated_items(
            [{"id": 1}],
            count=invalid_count,
            seed=0,
        )


@pytest.mark.parametrize("entrypoint", ["train.py", "eval.py"])
def test_legacy_entrypoint_cli_parses_without_augmentation_options(
    entrypoint: str,
) -> None:
    parser = load_entrypoint_parser(REPOSITORY_ROOT / entrypoint)

    args = parser.parse_args([])

    assert args.add_filter_ratio == 0.0
    assert args.naive_num == 1
    assert args.checkpoint_name is None


@pytest.mark.parametrize("option", ["--add-filter-ratio", "--add_filter_ratio"])
def test_filter_ratio_aliases_and_bounds(option: str) -> None:
    parser = load_entrypoint_parser(REPOSITORY_ROOT / "train.py")

    assert parser.parse_args([option, "0.25"]).add_filter_ratio == 0.25
    with pytest.raises(SystemExit):
        parser.parse_args([option, "-0.1"])
    with pytest.raises(SystemExit):
        parser.parse_args([option, "1.1"])


@pytest.mark.parametrize("value", ["0", "-1", "1.5"])
def test_naive_num_requires_positive_integer(value: str) -> None:
    parser = load_entrypoint_parser(REPOSITORY_ROOT / "train.py")

    with pytest.raises(SystemExit):
        parser.parse_args(["--naive-num", value])


def test_checkpoint_default_and_override_are_explicit() -> None:
    source = (REPOSITORY_ROOT / "train.py").read_text()

    assert "args.checkpoint_name or 'checkpoint_best'" in source
    assert "--checkpoint-name" in source


class FakeKGTokenizer:
    def __init__(self) -> None:
        self.qav_max_seq_len = None

    def encode_kvqa(self, **kwargs):
        return [[10, 11, -2, -2, 14, 15, 16, 17]], 6, 2, 5, 6

    def encode_kvaq(self, **kwargs):
        return [[20, 21, -2, -2, 24, 25, 26, 27]], 6, 2, 5, 6

    def encode_kqav(self, **kwargs):
        self.qav_max_seq_len = kwargs["max_seq_len"]
        return [[30, 31, 32, 33, -2, -2, 36]], 4, 1, 2


def load_kg_modules():
    package = types.ModuleType("dataloader")
    package.__path__ = [str(REPOSITORY_ROOT / "dataloader")]
    saved = {
        key: sys.modules.get(key)
        for key in (
            "dataloader",
            "dataloader.base_dataset_kg",
            "dataloader.dramaqa_kg",
            "dataloader.inquirer_paths",
        )
    }
    sys.modules["dataloader"] = package
    try:
        paths = load_module(
            "dataloader.inquirer_paths",
            REPOSITORY_ROOT / "dataloader" / "inquirer_paths.py",
        )
        sys.modules["dataloader.inquirer_paths"] = paths
        base = load_module(
            "dataloader.base_dataset_kg",
            REPOSITORY_ROOT / "dataloader" / "base_dataset_kg.py",
        )
        drama = load_module(
            "dataloader.dramaqa_kg",
            REPOSITORY_ROOT / "dataloader" / "dramaqa_kg.py",
        )
        return base, drama, saved
    except Exception:
        restore_modules(saved)
        raise


def restore_modules(saved: dict[str, object | None]) -> None:
    for key, value in saved.items():
        if value is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = value


def load_batch_collate():
    package_name = "quality_dataloader"
    package = types.ModuleType(package_name)
    package.__path__ = [str(REPOSITORY_ROOT / "dataloader")]
    sys.modules[package_name] = package
    for module_name, class_name in (
        ("nextqa", "NextQA"),
        ("dramaqa", "DramaQA"),
        ("dramaqa_kg", "DramaQA_KG"),
        ("star", "STAR"),
        ("vlep", "VLEP"),
        ("tvqa", "TVQA"),
        ("how2qa", "How2QA"),
    ):
        stub = types.ModuleType(f"{package_name}.{module_name}")
        setattr(stub, class_name, type(class_name, (), {}))
        sys.modules[stub.__name__] = stub
    return load_module(
        package_name,
        REPOSITORY_ROOT / "dataloader" / "__init__.py",
    ).batch_collate


def test_dramaqa_kg_item_matches_tokenizer_and_collate_contract() -> None:
    base, drama, saved = load_kg_modules()
    try:
        args = SimpleNamespace(max_feats=2, max_seq_len=10, rank=0)
        tokenizer = FakeKGTokenizer()
        dataset = drama.DramaQA_KG.__new__(drama.DramaQA_KG)
        base.BaseDataset_kg.__init__(dataset, args, tokenizer, "train")
        dataset.answer_mapping = {
            0: "(A)",
            1: "(B)",
            2: "(C)",
            3: "(D)",
            4: "(E)",
        }
        dataset.num_options = 5
        dataset.data = [
            {
                "vid": "clip0001",
                "que": "what happens",
                "answers": ["a", "b", "c", "d", "e"],
                "correct_idx": 0,
                "kg": {"subject": "event"},
            }
        ]
        dataset.features = {"clip0001": torch.ones(2, 768)}

        item = dataset[0]
        batch = load_batch_collate()([item])
    finally:
        restore_modules(saved)

    assert tokenizer.qav_max_seq_len == 10
    assert item["video"].shape == (2, 768)
    assert set(item["text_id"]) == {"vqa", "vaq", "qav"}
    assert all(value.shape == (1, 10) for value in item["text_id"].values())
    assert all(value.shape == (1, 10) for value in item["label"].values())
    assert item["video_start"] == {"vqa": 2, "vaq": 2, "qav": 4}
    assert all(value.shape == (2,) for value in item["video_index"].values())
    assert batch["video"].shape == (1, 2, 768)
    assert batch["text_id"]["vqa"].shape == (1, 1, 10)
    assert batch["video_index"]["qav"].shape == (1, 2)


def test_kg_padding_crops_the_same_context_span_for_every_choice() -> None:
    base, _, saved = load_kg_modules()
    try:
        dataset = base.BaseDataset_kg.__new__(base.BaseDataset_kg)
        dataset.args = SimpleNamespace(rank=1)
        dataset.max_seq_len = 8
        text_ids = [
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8]),
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        ]

        padded, prefix = dataset._get_padding_id(
            text_ids,
            prefix_index=7,
            context_start=2,
            context_end=6,
        )
    finally:
        restore_modules(saved)

    assert prefix == 4
    assert padded[0].tolist() == [0, 1, 2, 6, 7, 8, -1, -1]
    assert padded[1].tolist() == [0, 1, 2, 6, 7, 8, 9, 10]


@pytest.mark.parametrize(
    "loader_path",
    [
        "dataloader/dramaqa.py",
        "dataloader/star.py",
        "dataloader/tvqa.py",
    ],
)
def test_augmented_loaders_use_shared_filter_and_local_sampling(
    loader_path: str,
) -> None:
    source = (REPOSITORY_ROOT / loader_path).read_text()

    assert "filter_generated_items" in source
    assert "sample_generated_items" in source
    assert "seed=args.seed" in source
    assert "random.sample" not in source
