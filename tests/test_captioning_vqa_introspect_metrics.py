import importlib.util
import json
import sys
import types
from pathlib import Path


class _Registry:
    def __init__(self):
        self.paths = {"cache_root": "/unused", "repo_root": "/unused", "output_dir": "/unused"}

    def register_task(self, name):
        def decorator(cls):
            return cls
        return decorator

    def get_path(self, name):
        return self.paths[name]


REGISTRY = _Registry()


def _load_captioning_module():
    stub_names = [
        "lavis",
        "lavis.common",
        "lavis.tasks",
        "lavis.common.dist_utils",
        "lavis.common.registry",
        "lavis.tasks.base_task",
    ]
    previous_modules = {name: sys.modules.get(name) for name in stub_names}

    dist_utils = types.ModuleType("lavis.common.dist_utils")
    dist_utils.main_process = lambda fn: fn

    registry_module = types.ModuleType("lavis.common.registry")
    registry_module.registry = REGISTRY

    base_task_module = types.ModuleType("lavis.tasks.base_task")

    class _BaseTask:
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def save_result(*args, **kwargs):
            raise AssertionError("save_result is not used in this test")

    base_task_module.BaseTask = _BaseTask

    sys.modules["lavis"] = types.ModuleType("lavis")
    sys.modules["lavis.common"] = types.ModuleType("lavis.common")
    sys.modules["lavis.tasks"] = types.ModuleType("lavis.tasks")
    sys.modules["lavis.common.dist_utils"] = dist_utils
    sys.modules["lavis.common.registry"] = registry_module
    sys.modules["lavis.tasks.base_task"] = base_task_module

    module_path = Path("lavis/tasks/captioning.py")
    spec = importlib.util.spec_from_file_location("captioning_under_test", module_path)
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


def test_vqa_introspect_answer_metric_uses_normalized_exact_match(tmp_path):
    captioning = _load_captioning_module()
    result_file = tmp_path / "val_epoch0.json"
    result_file.write_text(
        json.dumps(
            [
                {"caption": "The yes.", "text_output": "yes"},
                {"caption": "no", "text_output": "No!"},
                {"caption": "left", "text_output": "right"},
            ]
        )
    )
    task = captioning.VQAIntrospectCaptionTask(
        num_beams=1,
        max_len=5,
        min_len=1,
        evaluate=False,
        metric="answer_exact_match",
    )

    metrics = task._report_metrics(str(result_file), "val")

    assert metrics == {"agg_metrics": 2 / 3, "answer_exact_match": 2 / 3}


def test_caption_valid_step_preserves_instance_id_for_duplicate_removal():
    captioning = _load_captioning_module()

    class _Model:
        def generate(self, samples, **kwargs):
            return ["answer"]

    task = captioning.CaptionTask(
        num_beams=1,
        max_len=5,
        min_len=1,
        evaluate=False,
    )
    results = task.valid_step(
        _Model(),
        {
            "image_id": [10],
            "main_question_id": [20],
            "instance_id": ["sample-1"],
            "text_input": ["question"],
            "text_output": ["answer"],
        },
    )

    assert results[0]["instance_id"] == "sample-1"
