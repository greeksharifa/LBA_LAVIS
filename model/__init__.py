from pathlib import Path
from typing import List, Any

import torch

from config.configs import Config
from model.models import *


MODEL_REGISTRY = {
    # "qwen2-vl-2b": Qwen2VL,
    # "qwen2-vl-7b": Qwen2VL,

    "qwen2.5-vl-7b": Qwen2_5VL,
    "qwen3-vl-8b": Qwen2_5VL,
    # "qwen2.5-vl-3b": Qwen2_5VL,
    
    # "GPT4o": GPT4o,
}


def get_model(cfg: Config) -> C2RFramework:
    model_name = cfg.model_cfg.model_name
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in registry")
    
    # model_id = cfg.model_cfg.model_id
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(cfg)

    return model

