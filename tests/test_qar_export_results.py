import importlib.util
import json
import sys
import types

import pytest
from pathlib import Path


def _install_stub_modules():
    stub_names = [
        "lavis",
        "lavis.tasks",
        "lavis.common",
        "lavis.common.dist_utils",
        "lavis.common.config",
        "lavis.common.logger",
        "lavis.common.optims",
        "lavis.common.registry",
        "lavis.common.utils",
        "lavis.datasets",
        "lavis.datasets.builders",
        "lavis.models",
        "lavis.processors",
        "lavis.runners",
        "subqa",
        "subqa.vqa_introspect_export",
    ]
    previous_modules = {name: sys.modules.get(name) for name in stub_names}

    lavis = types.ModuleType("lavis")
    sys.modules["lavis"] = lavis

    tasks_module = types.ModuleType("lavis.tasks")
    common_module = types.ModuleType("lavis.common")
    dist_utils = types.ModuleType("lavis.common.dist_utils")
    dist_utils.get_rank = lambda: 0
    dist_utils.init_distributed_mode = lambda *args, **kwargs: None
    config_module = types.ModuleType("lavis.common.config")
    config_module.Config = object
    logger_module = types.ModuleType("lavis.common.logger")
    logger_module.setup_logger = lambda *args, **kwargs: None
    optims_module = types.ModuleType("lavis.common.optims")
    optims_module.LinearWarmupCosineLRScheduler = object
    optims_module.LinearWarmupStepLRScheduler = object
    registry_module = types.ModuleType("lavis.common.registry")
    registry_module.registry = object()
    utils_module = types.ModuleType("lavis.common.utils")
    utils_module.now = lambda: "now"

    sys.modules["lavis.tasks"] = tasks_module
    sys.modules["lavis.common"] = common_module
    sys.modules["lavis.common.dist_utils"] = dist_utils
    sys.modules["lavis.common.config"] = config_module
    sys.modules["lavis.common.logger"] = logger_module
    sys.modules["lavis.common.optims"] = optims_module
    sys.modules["lavis.common.registry"] = registry_module
    sys.modules["lavis.common.utils"] = utils_module

    for name in [
        "lavis.datasets",
        "lavis.datasets.builders",
        "lavis.models",
        "lavis.processors",
        "lavis.runners",
    ]:
        sys.modules[name] = types.ModuleType(name)

    subqa_module = types.ModuleType("subqa")
    export_module = types.ModuleType("subqa.vqa_introspect_export")
    export_module.build_compact_subqa_mapping = lambda results, dedupe=False: {
        "dedupe": dedupe,
        "count": len(results),
    }
    sys.modules["subqa"] = subqa_module
    sys.modules["subqa.vqa_introspect_export"] = export_module
    return previous_modules


def _load_test_qar_module():
    previous_modules = _install_stub_modules()
    module_path = Path("test_qar.py")
    spec = importlib.util.spec_from_file_location("test_qar_under_test", module_path)
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


def test_export_compact_results_rejects_captioning_epoch_result(tmp_path):
    module = _load_test_qar_module()
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    (result_dir / "val_epochbest.json").write_text(
        json.dumps([{"main_question_id": 1, "caption": "why?"}])
    )
    with pytest.raises(FileNotFoundError, match="val_vqa_result.json"):
        module._export_compact_results(str(result_dir), "val", str(tmp_path / "compact"), "unit")


def test_qar_dataset_options_does_not_add_caption_dataset(tmp_path):
    module = _load_test_qar_module()
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("datasets:\n  vqa_introspect_test: {}\nrun:\n  task: vqa_introspect_test_task\n")
    assert module._qar_dataset_options(str(cfg), image_root="/images") == [
        "datasets.vqa_introspect_test.build_info.images.storage=/images"
    ]


def test_qar_dataset_options_rejects_caption_config_before_model_load(tmp_path):
    module = _load_test_qar_module()
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("datasets:\n  vqa_introspect_qar_caption: {}\nrun:\n  task: vqa_introspect_captioning\n")
    with pytest.raises(ValueError, match="vqa_introspect_test"):
        module._qar_dataset_options(str(cfg))
