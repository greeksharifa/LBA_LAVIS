"""Stable, testable vLLM process and engine configuration."""

import os
from typing import Callable, Dict, Optional, Union


def configure_vllm_environment() -> None:
    """Set safe vLLM defaults without replacing explicit user choices."""
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


configure_vllm_environment()


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

    limit_config = dataset_cfg.get("limit_mm_per_prompt", None)
    if limit_config is not None:
        modality = dataset_cfg.data_type
        kwargs["limit_mm_per_prompt"] = {modality: limit_config[modality]}
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
    if visible_device_count is None:
        visible_count = int(_cuda_device_count())
    elif callable(visible_device_count):
        visible_count = int(visible_device_count())
    else:
        visible_count = int(visible_device_count)
    tensor_parallel_size = int(tensor_parallel_size)
    if tensor_parallel_size != visible_count:
        raise ValueError(
            "vLLM tensor_parallel_size="
            f"{tensor_parallel_size} does not match visible CUDA device count="
            f"{visible_count}. Set model.tensor_parallel_size to the number of "
            "GPUs visible to this process."
        )
    return tensor_parallel_size
