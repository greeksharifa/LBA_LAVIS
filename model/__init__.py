"""Model factory with lazy runtime imports.

Keeping this package initializer lightweight lets CPU-only tooling import
``model.vllm_config`` without importing torch or vLLM.
"""

from importlib import import_module


_MODEL_CLASS_NAMES = {
    "qwen2.5-vl-7b": "Qwen2_5VL",
    "qwen3-vl-8b": "Qwen2_5VL",
}

__all__ = ["C2RFramework", "Qwen2_5VL", "MODEL_REGISTRY", "get_model"]


def _models_module():
    return import_module("model.models")


def get_model(cfg):
    model_name = cfg.model_cfg.model_name
    class_name = _MODEL_CLASS_NAMES.get(model_name)
    if class_name is None:
        raise ValueError(f"Model {model_name} not found in registry")
    model_class = getattr(_models_module(), class_name)
    return model_class(cfg)


def __getattr__(name):
    if name in ("C2RFramework", "Qwen2_5VL"):
        return getattr(_models_module(), name)
    if name == "MODEL_REGISTRY":
        models = _models_module()
        return {
            model_name: getattr(models, class_name)
            for model_name, class_name in _MODEL_CLASS_NAMES.items()
        }
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
