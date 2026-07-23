import torch

from lavis.runners.runner_base import RunnerBase
from lavis.common.registry import registry


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(1, 1)

    @property
    def device(self):
        return self.proj.weight.device


class _DummyCfg:
    class _RunCfg(dict):
        def __getattr__(self, key):
            return self[key]

        def get(self, key, default=None):
            return super().get(key, default)

    def __init__(self, output_dir="tmp_runner_test_output", device="cpu"):
        self.run_cfg = self._RunCfg(
            output_dir=output_dir,
            device=device,
            distributed=False,
        )


def test_runner_base_model_returns_model_when_already_on_target_device():
    registry.mapping["paths"].pop("result_dir", None)
    registry.mapping["paths"].pop("output_dir", None)

    cfg = _DummyCfg()
    runner = RunnerBase(
        cfg=cfg,
        task=object(),
        model=_DummyModel(),
        datasets={},
        job_id="runner-model-device-test",
    )

    assert runner.model is runner._model
