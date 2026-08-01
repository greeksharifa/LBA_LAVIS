"""Stable, testable vLLM process and engine configuration."""

import os
from typing import Callable, Dict, Optional, Union


def configure_vllm_environment() -> None:
    """Set safe vLLM defaults without replacing explicit user choices."""
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


configure_vllm_environment()


def patch_zero_count_dummy_video_allocation(builder_cls=None) -> None:
    """Work around vLLM 0.8.2 allocating a dummy video for count zero.

    Qwen2-VL's dummy-input builder constructs the maximum-size video array
    before multiplying the result list by ``num_videos``. On image-only jobs
    that count is zero, but the discarded allocation can still consume tens of
    GiB per tensor-parallel process.
    """
    if builder_cls is None:
        from vllm.multimodal.profiling import BaseDummyInputsBuilder

        builder_cls = BaseDummyInputsBuilder

    original = builder_cls._get_dummy_videos
    if getattr(original, "_lba_skips_zero_video", False):
        return

    def _skip_zero_video(
        self, *, width: int, height: int, num_frames: int, num_videos: int
    ):
        if num_videos == 0:
            return []
        return original(
            self,
            width=width,
            height=height,
            num_frames=num_frames,
            num_videos=num_videos,
        )

    _skip_zero_video._lba_skips_zero_video = True
    builder_cls._get_dummy_videos = _skip_zero_video


def build_engine_kwargs(cfg) -> Dict[str, object]:
    """Build the kwargs passed to ``vllm.LLM`` from application config."""
    model_cfg = cfg.model_cfg if hasattr(cfg, "model_cfg") else cfg.model
    dataset_cfg = cfg.dataset_cfg if hasattr(cfg, "dataset_cfg") else cfg.dataset
    kwargs = {
        "model": model_cfg.model_id,
        "max_model_len": int(model_cfg.max_model_len),
        "max_num_seqs": int(model_cfg.max_num_seqs),
        "tensor_parallel_size": int(model_cfg.tensor_parallel_size),
        "gpu_memory_utilization": float(model_cfg.gpu_memory_utilization),
        "swap_space": float(model_cfg.swap_space),
        "enforce_eager": bool(model_cfg.enforce_eager),
        "mm_processor_kwargs": {
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
    }
    if os.environ.get("VLLM_USE_V1", "0") == "0":
        kwargs["worker_cls"] = "model.vllm_worker.ZeroCountVideoSafeWorker"

    limit_config = dataset_cfg.get("limit_mm_per_prompt", None)
    if limit_config is not None:
        modality = dataset_cfg.data_type
        if modality not in limit_config:
            raise ValueError(
                "dataset.limit_mm_per_prompt must configure its active "
                f"modality: {modality}"
            )
        kwargs["limit_mm_per_prompt"] = {
            str(name): int(limit) for name, limit in limit_config.items()
        }
    return kwargs


def _cuda_device_count() -> int:
    import torch

    return torch.cuda.device_count()


def validate_tensor_parallel_size(
    tensor_parallel_size: int,
    *,
    visible_device_count: Optional[Union[int, Callable[[], int]]] = None,
) -> int:
    """Fail before model construction if TP does not match visible CUDA GPUs."""
    tensor_parallel_size = int(tensor_parallel_size)
    if tensor_parallel_size < 1:
        raise ValueError("vLLM tensor_parallel_size must be at least 1")
    if visible_device_count is None:
        visible_count = int(_cuda_device_count())
    elif callable(visible_device_count):
        visible_count = int(visible_device_count())
    else:
        visible_count = int(visible_device_count)
    if visible_count < 1:
        raise ValueError("vLLM visible CUDA device count must be at least 1")
    if tensor_parallel_size != visible_count:
        raise ValueError(
            "vLLM tensor_parallel_size="
            f"{tensor_parallel_size} does not match visible CUDA device count="
            f"{visible_count}. Set model.tensor_parallel_size to the number of "
            "GPUs visible to this process."
        )
    return tensor_parallel_size
