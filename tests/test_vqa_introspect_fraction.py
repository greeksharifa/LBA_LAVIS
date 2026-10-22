import importlib.util
import json
import random
import sys
import types
from pathlib import Path


class _CaptionDataset:
    def _add_instance_ids(self):
        for idx, ann in enumerate(self.annotation):
            ann["instance_id"] = idx


class _CaptionEvalDataset(_CaptionDataset):
    pass


def _load_vqa_introspect_module():
    stub_names = [
        "lavis",
        "lavis.datasets",
        "lavis.datasets.datasets",
        "lavis.datasets.datasets.caption_datasets",
        "lavis.common",
        "lavis.common.registry",
        "colors",
    ]
    previous_modules = {name: sys.modules.get(name) for name in stub_names}

    caption_module = types.ModuleType("lavis.datasets.datasets.caption_datasets")
    caption_module.CaptionDataset = _CaptionDataset
    caption_module.CaptionEvalDataset = _CaptionEvalDataset

    registry_module = types.ModuleType("lavis.common.registry")
    registry_module.registry = object()

    colors_module = types.ModuleType("colors")
    colors_module.Colors = types.SimpleNamespace(BRIGHT_MAGENTA="", RESET="")
    colors_module.print_sample = lambda *args, **kwargs: None

    sys.modules["lavis"] = types.ModuleType("lavis")
    sys.modules["lavis.datasets"] = types.ModuleType("lavis.datasets")
    sys.modules["lavis.datasets.datasets"] = types.ModuleType("lavis.datasets.datasets")
    sys.modules["lavis.datasets.datasets.caption_datasets"] = caption_module
    sys.modules["lavis.common"] = types.ModuleType("lavis.common")
    sys.modules["lavis.common.registry"] = registry_module
    sys.modules["colors"] = colors_module

    module_path = Path("lavis/datasets/datasets/vqa_introspect_datasets.py")
    spec = importlib.util.spec_from_file_location("vqa_introspect_datasets_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


vqa_introspect = _load_vqa_introspect_module()
_apply_sample_fraction = vqa_introspect._apply_sample_fraction
_get_text_input_output = vqa_introspect._get_text_input_output
VQAIntrospectQARCapEvalDataset = vqa_introspect.VQAIntrospectQARCapEvalDataset
VQAIntrospectQARCapDataset = vqa_introspect.VQAIntrospectQARCapDataset


def test_fraction_counts_match_sq_instructblip_plan():
    records = list(range(55799))

    assert len(_apply_sample_fraction(records, 1 / 8, sample_seed=42)) == 6974
    assert len(_apply_sample_fraction(records, 1 / 4, sample_seed=42)) == 13949
    assert len(_apply_sample_fraction(records, 1 / 2, sample_seed=42)) == 27899
    assert len(_apply_sample_fraction(records, 1, sample_seed=42)) == 55799


def test_fraction_splits_are_deterministic_prefixes():
    records = list(range(55799))

    small = _apply_sample_fraction(records, 1 / 8, sample_seed=42)
    medium = _apply_sample_fraction(records, 1 / 4, sample_seed=42)
    large = _apply_sample_fraction(records, 1 / 2, sample_seed=42)
    full = _apply_sample_fraction(records, 1, sample_seed=42)

    assert small == medium[: len(small)]
    assert medium == large[: len(medium)]
    assert large == full[: len(large)]


def _write_eval_annotation(path, count):
    path.write_text(
        json.dumps(
            {
                str(1000 + idx): {
                    "image_id": idx,
                    "reasoning_question": f"question {idx}?",
                    "reasoning_answer_most_common": f"answer {idx}",
                    "introspect": [
                        {
                            "pred_q_type": "reasoning",
                            "sub_qa": [
                                {
                                    "sub_question": f"sub question {idx}?",
                                    "sub_answer": f"sub answer {idx}",
                                }
                            ],
                        }
                    ],
                }
                for idx in range(count)
            }
        )
    )


def test_eval_dataset_uses_full_val_when_eval_max_samples_is_none(tmp_path):
    ann_path = tmp_path / "ann.json"
    _write_eval_annotation(ann_path, 7)

    dataset = VQAIntrospectQARCapEvalDataset(
        vis_processor=None,
        text_processor=None,
        vis_root="/unused",
        ann_paths=[str(ann_path)],
        prompt_type="Reasoner",
        eval_max_samples=None,
    )

    assert len(dataset.annotation) == 7


def test_eval_dataset_applies_eval_max_samples_cap(tmp_path):
    ann_path = tmp_path / "ann.json"
    _write_eval_annotation(ann_path, 7)

    dataset = VQAIntrospectQARCapEvalDataset(
        vis_processor=None,
        text_processor=None,
        vis_root="/unused",
        ann_paths=[str(ann_path)],
        prompt_type="Questioner_MultipleSubQ",
        eval_max_samples=3,
    )

    assert len(dataset.annotation) == 3


def test_eval_max_samples_cap_applies_across_all_annotation_files(tmp_path):
    ann_path_a = tmp_path / "ann_a.json"
    ann_path_b = tmp_path / "ann_b.json"
    _write_eval_annotation(ann_path_a, 3)
    _write_eval_annotation(ann_path_b, 3)

    dataset = VQAIntrospectQARCapEvalDataset(
        vis_processor=None,
        text_processor=None,
        vis_root="/unused",
        ann_paths=[str(ann_path_a), str(ann_path_b)],
        prompt_type="Reasoner",
        eval_max_samples=2,
    )

    assert len(dataset.annotation) == 2


def test_questioner_eval_prompt_is_deterministic_and_preserves_target_order():
    ann = {
        "main_question": "what is happening?",
        "main_answer": "yes",
        "sub_question_list": ["first?", "second?", "third?", "fourth?"],
        "sub_answer_list": ["one", "two", "three", "four"],
    }

    first = _get_text_input_output(
        "Questioner_MultipleSubQ", ann, "eval", rng=random.Random(7)
    )
    second = _get_text_input_output(
        "Questioner_MultipleSubQ", ann, "eval", rng=random.Random(7)
    )

    assert first == second
    target_questions = first[1].split("\n")
    assert target_questions == [
        question
        for question in ann["sub_question_list"]
        if question in target_questions
    ]

def test_questioner_prompt_type_alias_uses_multiple_subq_prompt():
    ann = {
        "main_question": "what is happening?",
        "main_answer": "yes",
        "sub_question_list": ["first?", "second?", "third?", "fourth?"],
        "sub_answer_list": ["one", "two", "three", "four"],
    }

    alias_result = _get_text_input_output("Questioner", ann, "train", rng=random.Random(7))
    canonical_result = _get_text_input_output(
        "Questioner_MultipleSubQ", ann, "train", rng=random.Random(7)
    )

    assert alias_result == canonical_result


def test_train_dataset_accepts_questioner_prompt_type_alias(tmp_path):
    ann_path = tmp_path / "ann.json"
    _write_eval_annotation(ann_path, 3)

    dataset = VQAIntrospectQARCapDataset(
        vis_processor=None,
        text_processor=None,
        vis_root="/unused",
        ann_paths=[str(ann_path)],
        prompt_type="Questioner",
        sample_fraction=1,
    )

    assert len(dataset.annotation) == 3
