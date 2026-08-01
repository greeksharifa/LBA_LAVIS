import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf


class VllmConfigTests(unittest.TestCase):
    def test_importing_config_does_not_import_vllm_or_torch(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import model.vllm_config; "
                    "assert 'vllm' not in sys.modules; "
                    "assert 'torch' not in sys.modules"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(0, result.returncode, result.stderr)

    def test_importing_mutable_registry_does_not_import_vllm_or_torch(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; from model import MODEL_REGISTRY; "
                    "assert 'vllm' not in sys.modules; "
                    "assert 'torch' not in sys.modules"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(0, result.returncode, result.stderr)

    def test_mutable_registry_extensions_are_used_by_model_factory(self):
        from model import MODEL_REGISTRY, get_model

        class FixtureModel:
            def __init__(self, cfg):
                self.cfg = cfg

        cfg = SimpleNamespace(
            model_cfg=SimpleNamespace(model_name="fixture-model")
        )
        MODEL_REGISTRY["fixture-model"] = FixtureModel
        try:
            model = get_model(cfg)
            self.assertIsInstance(model, FixtureModel)
            self.assertIs(cfg, model.cfg)
        finally:
            del MODEL_REGISTRY["fixture-model"]

    def test_environment_defaults_preserve_user_overrides(self):
        from model.vllm_config import configure_vllm_environment

        with patch.dict(os.environ, {}, clear=True):
            configure_vllm_environment()
            self.assertEqual("0", os.environ["VLLM_USE_V1"])
            self.assertEqual("spawn", os.environ["VLLM_WORKER_MULTIPROC_METHOD"])

        with patch.dict(
            os.environ,
            {
                "VLLM_USE_V1": "1",
                "VLLM_WORKER_MULTIPROC_METHOD": "fork",
            },
            clear=True,
        ):
            configure_vllm_environment()
            self.assertEqual("1", os.environ["VLLM_USE_V1"])
            self.assertEqual("fork", os.environ["VLLM_WORKER_MULTIPROC_METHOD"])

    def test_engine_kwargs_include_all_runtime_controls(self):
        from model.vllm_config import build_engine_kwargs

        cfg = OmegaConf.create(
            {
                "model": {
                    "model_id": "fixture/model",
                    "max_model_len": 8192,
                    "max_num_seqs": 17,
                    "tensor_parallel_size": 3,
                    "gpu_memory_utilization": 0.81,
                    "swap_space": 7,
                    "enforce_eager": False,
                },
                "dataset": {"data_type": "image"},
            }
        )

        kwargs = build_engine_kwargs(cfg)

        self.assertEqual(8192, kwargs["max_model_len"])
        self.assertEqual(17, kwargs["max_num_seqs"])
        self.assertEqual(3, kwargs["tensor_parallel_size"])
        self.assertEqual(0.81, kwargs["gpu_memory_utilization"])
        self.assertEqual(7, kwargs["swap_space"])
        self.assertIs(False, kwargs["enforce_eager"])
        self.assertEqual(
            "model.vllm_worker.ZeroCountVideoSafeWorker",
            kwargs["worker_cls"],
        )

    def test_engine_kwargs_preserve_zero_limits_for_unused_modalities(self):
        from model.vllm_config import build_engine_kwargs

        cfg = OmegaConf.create(
            {
                "model": {
                    "model_id": "fixture/model",
                    "max_model_len": 8192,
                    "max_num_seqs": 17,
                    "tensor_parallel_size": 1,
                    "gpu_memory_utilization": 0.81,
                    "swap_space": 7,
                    "enforce_eager": True,
                },
                "dataset": {
                    "data_type": "image",
                    "limit_mm_per_prompt": {"image": 7, "video": 0},
                },
            }
        )

        kwargs = build_engine_kwargs(cfg)

        self.assertEqual(
            {"image": 7, "video": 0},
            kwargs["limit_mm_per_prompt"],
        )

    def test_v1_keeps_vllm_automatic_worker_selection(self):
        from model.vllm_config import build_engine_kwargs

        cfg = OmegaConf.create(
            {
                "model": {
                    "model_id": "fixture/model",
                    "max_model_len": 8192,
                    "max_num_seqs": 17,
                    "tensor_parallel_size": 1,
                    "gpu_memory_utilization": 0.81,
                    "swap_space": 7,
                    "enforce_eager": True,
                },
                "dataset": {"data_type": "image"},
            }
        )

        with patch.dict(os.environ, {"VLLM_USE_V1": "1"}):
            kwargs = build_engine_kwargs(cfg)

        self.assertNotIn("worker_cls", kwargs)

    def test_mmmu_disables_unused_video_inputs(self):
        dataset_cfg = OmegaConf.load("config/datasets/MMMU.yaml")["dataset"]

        self.assertEqual(7, dataset_cfg.limit_mm_per_prompt.image)
        self.assertEqual(0, dataset_cfg.limit_mm_per_prompt.video)

    def test_zero_count_dummy_video_patch_skips_the_vllm_allocation(self):
        from model.vllm_config import patch_zero_count_dummy_video_allocation

        class FakeDummyInputsBuilder:
            calls = 0

            def _get_dummy_videos(
                self, *, width, height, num_frames, num_videos
            ):
                type(self).calls += 1
                return [object()] * num_videos

        patch_zero_count_dummy_video_allocation(FakeDummyInputsBuilder)
        patch_zero_count_dummy_video_allocation(FakeDummyInputsBuilder)
        builder = FakeDummyInputsBuilder()

        self.assertEqual(
            [],
            builder._get_dummy_videos(
                width=1280, height=1280, num_frames=1024, num_videos=0
            ),
        )
        self.assertEqual(0, FakeDummyInputsBuilder.calls)
        self.assertEqual(
            2,
            len(
                builder._get_dummy_videos(
                    width=16, height=16, num_frames=2, num_videos=2
                )
            ),
        )
        self.assertEqual(1, FakeDummyInputsBuilder.calls)

    def test_custom_worker_applies_patch_inside_spawned_process(self):
        from model.vllm_worker import ZeroCountVideoSafeWorker
        from vllm.worker.worker import Worker

        worker = object.__new__(ZeroCountVideoSafeWorker)
        with patch(
            "model.vllm_worker.patch_zero_count_dummy_video_allocation"
        ) as apply_patch, patch.object(Worker, "__init__", return_value=None) as parent_init:
            ZeroCountVideoSafeWorker.__init__(worker, fixture=True)

        apply_patch.assert_called_once_with()
        parent_init.assert_called_once_with(fixture=True)

    def test_tensor_parallel_preflight_rejects_visible_device_mismatch(self):
        from model.vllm_config import validate_tensor_parallel_size

        with self.assertRaisesRegex(
            ValueError,
            "tensor_parallel_size=4.*visible CUDA device count=2",
        ):
            validate_tensor_parallel_size(4, visible_device_count=lambda: 2)

    def test_tensor_parallel_preflight_accepts_matching_device_count(self):
        from model.vllm_config import validate_tensor_parallel_size

        self.assertEqual(
            4,
            validate_tensor_parallel_size(4, visible_device_count=4),
        )

    def test_tensor_parallel_preflight_rejects_non_positive_tp_before_gpu_query(self):
        from model.vllm_config import validate_tensor_parallel_size

        with self.assertRaisesRegex(ValueError, "tensor_parallel_size must be at least 1"):
            validate_tensor_parallel_size(
                0,
                visible_device_count=lambda: self.fail("GPU count must not be queried"),
            )

    def test_tensor_parallel_preflight_rejects_no_visible_gpus(self):
        from model.vllm_config import validate_tensor_parallel_size

        with self.assertRaisesRegex(ValueError, "visible CUDA device count must be at least 1"):
            validate_tensor_parallel_size(1, visible_device_count=0)


if __name__ == "__main__":
    unittest.main()
