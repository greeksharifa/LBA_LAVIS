"""vLLM worker compatibility hooks for spawned tensor-parallel ranks."""

from model.vllm_config import patch_zero_count_dummy_video_allocation
from vllm.worker.worker import Worker


class ZeroCountVideoSafeWorker(Worker):
    """Apply the Qwen2-VL zero-video fix inside every spawned worker."""

    def __init__(self, *args, **kwargs):
        patch_zero_count_dummy_video_allocation()
        super().__init__(*args, **kwargs)
