import importlib.util
import sys
import types
from pathlib import Path

import torch


class _Registry:
    def register_lr_scheduler(self, name):
        def decorator(cls):
            return cls
        return decorator


def _load_optims_module():
    stub_names = ["lavis", "lavis.common", "lavis.common.registry"]
    previous_modules = {name: sys.modules.get(name) for name in stub_names}

    registry_module = types.ModuleType("lavis.common.registry")
    registry_module.registry = _Registry()
    sys.modules["lavis"] = types.ModuleType("lavis")
    sys.modules["lavis.common"] = types.ModuleType("lavis.common")
    sys.modules["lavis.common.registry"] = registry_module

    module_path = Path("lavis/common/optims.py")
    spec = importlib.util.spec_from_file_location("optims_under_test", module_path)
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


def test_cosine_warmup_uses_global_step_across_epoch_boundary():
    optims = _load_optims_module()
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([param], lr=0.0)
    scheduler = optims.LinearWarmupCosineLRScheduler(
        optimizer=optimizer,
        max_epoch=10,
        min_lr=0.0,
        init_lr=1.0,
        warmup_start_lr=0.0,
        warmup_steps=8,
    )

    scheduler.step(cur_epoch=1, cur_step=0, num_iters_per_epoch=5)

    assert optimizer.param_groups[0]["lr"] == 5 / 8
