from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class StubBaseDataset:
    pass


class Column:
    def __init__(self, value):
        self.values = [value]


def load_how2qa_module():
    package = types.ModuleType("how2qa_test_dataloader")
    package.__path__ = [str(REPOSITORY_ROOT / "dataloader")]
    base = types.ModuleType(f"{package.__name__}.base_dataset")
    base.BaseDataset = StubBaseDataset
    paths = types.ModuleType(f"{package.__name__}.inquirer_paths")
    paths.resolve_dataset_path = lambda *args, **kwargs: None
    pandas = types.ModuleType("pandas")
    pandas.read_csv = lambda *args, **kwargs: None
    saved = {
        key: sys.modules.get(key)
        for key in (package.__name__, base.__name__, paths.__name__, "pandas")
    }
    sys.modules[package.__name__] = package
    sys.modules[base.__name__] = base
    sys.modules[paths.__name__] = paths
    sys.modules["pandas"] = pandas

    module_name = f"{package.__name__}.how2qa"
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPOSITORY_ROOT / "dataloader" / "how2qa.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        restore_modules(saved, module_name)
        raise
    return module, saved, module_name


def restore_modules(saved, module_name):
    sys.modules.pop(module_name, None)
    for key, value in saved.items():
        if value is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = value


def build_dataset(module, start, end):
    dataset = module.How2QA.__new__(module.How2QA)
    dataset.data = {
        "video_id": Column("video-1"),
        "answer_id": Column(0),
        "question": Column("what happens"),
        "a0": Column("first"),
        "a1": Column("second"),
        "a2": Column("third"),
        "a3": Column("fourth"),
        "start": Column(start),
        "end": Column(end),
    }
    dataset.answer_mapping = {0: "(A)", 1: "(B)", 2: "(C)", 3: "(D)"}
    dataset.num_options = 4
    dataset.features = {
        "video-1": torch.tensor(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
        )
    }
    dataset.features_dim = 2
    dataset.max_feats = 4
    dataset._get_text_token = lambda text, answer: (
        {},
        {},
        {},
        {},
        {},
    )

    video_calls = []
    get_video = module.How2QA._get_video.__get__(dataset, module.How2QA)

    def recording_get_video(video_id, normalized_start, normalized_end):
        video_calls.append((video_id, normalized_start, normalized_end))
        return get_video(video_id, normalized_start, normalized_end)

    dataset._get_video = recording_get_video
    return dataset, video_calls


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (None, None),
        (float("nan"), float("nan")),
        (float("inf"), float("-inf")),
    ],
)
def test_missing_or_nonfinite_timestamps_use_full_feature_fallback(
    start,
    end,
) -> None:
    module, saved, module_name = load_how2qa_module()
    try:
        dataset, video_calls = build_dataset(module, start, end)

        item = dataset[0]
    finally:
        restore_modules(saved, module_name)

    assert video_calls == [("video-1", None, None)]
    assert item["video_len"] == 3
    torch.testing.assert_close(
        item["video"],
        torch.tensor(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [0.0, 0.0]]
        ),
    )


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (None, 1.4),
        (0.6, float("nan")),
    ],
)
def test_mixed_missing_timestamps_disable_segment_slicing(start, end) -> None:
    module, saved, module_name = load_how2qa_module()
    try:
        dataset, video_calls = build_dataset(module, start, end)

        item = dataset[0]
    finally:
        restore_modules(saved, module_name)

    assert video_calls == [("video-1", None, None)]
    assert item["video_len"] == 3


def test_finite_timestamps_are_rounded_before_segment_slicing() -> None:
    module, saved, module_name = load_how2qa_module()
    try:
        dataset, video_calls = build_dataset(module, 0.6, 1.6)

        item = dataset[0]
    finally:
        restore_modules(saved, module_name)

    assert video_calls == [("video-1", 1, 2)]
    assert item["video_len"] == 2
    torch.testing.assert_close(
        item["video"],
        torch.tensor(
            [[2.0, 20.0], [3.0, 30.0], [0.0, 0.0], [0.0, 0.0]]
        ),
    )
