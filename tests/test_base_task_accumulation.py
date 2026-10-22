import importlib.util
import sys
import types
from pathlib import Path

import torch


class _Meter:
    def __init__(self):
        self.global_avg = 0.0


class _MetricLogger:
    def __init__(self, delimiter="  "):
        self.meters = {}

    def add_meter(self, name, meter):
        self.meters[name] = _Meter()

    def log_every(self, iterable, log_freq, header):
        return iterable

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.meters:
                self.meters[key] = _Meter()
            if torch.is_tensor(value):
                value = value.detach().float().item()
            self.meters[key].global_avg = float(value)

    def synchronize_between_processes(self):
        pass

    def global_avg(self):
        return ""


class _SmoothedValue:
    def __init__(self, *args, **kwargs):
        self.global_avg = 0.0


def _load_base_task_module():
    stub_names = [
        "lavis",
        "lavis.common",
        "lavis.datasets",
        "colors",
        "lavis.common.dist_utils",
        "lavis.common.logger",
        "lavis.common.registry",
        "lavis.datasets.data_utils",
    ]
    previous_modules = {name: sys.modules.get(name) for name in stub_names}

    colors_module = types.ModuleType("colors")
    colors_module.Colors = types.SimpleNamespace(BRIGHT_CYAN="", RESET="")
    colors_module.print_sample = lambda *args, **kwargs: None

    dist_utils = types.ModuleType("lavis.common.dist_utils")
    dist_utils.get_rank = lambda: 0
    dist_utils.get_world_size = lambda: 1
    dist_utils.is_main_process = lambda: True
    dist_utils.is_dist_avail_and_initialized = lambda: False

    logger_module = types.ModuleType("lavis.common.logger")
    logger_module.MetricLogger = _MetricLogger
    logger_module.SmoothedValue = _SmoothedValue

    registry_module = types.ModuleType("lavis.common.registry")
    registry_module.registry = object()

    data_utils = types.ModuleType("lavis.datasets.data_utils")
    data_utils.prepare_sample = lambda samples, cuda_enabled=False: samples

    sys.modules["lavis"] = types.ModuleType("lavis")
    sys.modules["lavis.common"] = types.ModuleType("lavis.common")
    sys.modules["lavis.datasets"] = types.ModuleType("lavis.datasets")
    sys.modules["colors"] = colors_module
    sys.modules["lavis.common.dist_utils"] = dist_utils
    sys.modules["lavis.common.logger"] = logger_module
    sys.modules["lavis.common.registry"] = registry_module
    sys.modules["lavis.datasets.data_utils"] = data_utils

    module_path = Path("lavis/tasks/base_task.py")
    spec = importlib.util.spec_from_file_location("base_task_under_test", module_path)
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


def test_train_inner_loop_flushes_final_partial_accumulation():
    base_task = _load_base_task_module()
    task = base_task.BaseTask()
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    step_calls = 0
    original_step = optimizer.step

    def counted_step(*args, **kwargs):
        nonlocal step_calls
        step_calls += 1
        return original_step(*args, **kwargs)

    optimizer.step = counted_step

    class _Scheduler:
        def step(self, cur_epoch, cur_step, **kwargs):
            pass

    def train_step(model, samples):
        loss = (model(samples["x"]) - samples["y"]).pow(2).mean()
        return loss, {"loss": loss.detach()}

    task.train_step = train_step
    data_loader = iter(
        {"x": torch.ones(1, 1), "y": torch.ones(1, 1)} for _ in range(3)
    )

    task._train_inner_loop(
        epoch=0,
        iters_per_epoch=3,
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        lr_scheduler=_Scheduler(),
        cuda_enabled=False,
        accum_grad_iters=2,
    )

    assert step_calls == 2
