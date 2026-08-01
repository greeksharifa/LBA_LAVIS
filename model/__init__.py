"""Model factory with lazy runtime imports.

Keeping this package initializer lightweight lets CPU-only tooling import
``model.vllm_config`` without importing torch or vLLM.
"""

from collections.abc import MutableMapping
from importlib import import_module


class _LazyModelClass:
    def __init__(self, class_name):
        self.class_name = class_name


class LazyModelRegistry(MutableMapping):
    """Mutable registry whose built-ins resolve only when accessed."""

    def __init__(self, builtins):
        self._entries = {
            model_name: _LazyModelClass(class_name)
            for model_name, class_name in builtins.items()
        }

    def __getitem__(self, model_name):
        value = self._entries[model_name]
        if isinstance(value, _LazyModelClass):
            value = getattr(_models_module(), value.class_name)
            self._entries[model_name] = value
        return value

    def __setitem__(self, model_name, model_class):
        self._entries[model_name] = model_class

    def __delitem__(self, model_name):
        del self._entries[model_name]

    def __iter__(self):
        return iter(self._entries)

    def __len__(self):
        return len(self._entries)


MODEL_REGISTRY = LazyModelRegistry(
    {
        "qwen2.5-vl-7b": "Qwen2_5VL",
        "qwen3-vl-8b": "Qwen2_5VL",
    }
)

__all__ = ["C2RFramework", "Qwen2_5VL", "MODEL_REGISTRY", "get_model"]


def _models_module():
    return import_module("model.models")


def get_model(cfg):
    model_name = cfg.model_cfg.model_name
    try:
        model_class = MODEL_REGISTRY[model_name]
    except KeyError:
        raise ValueError(f"Model {model_name} not found in registry")
    return model_class(cfg)


def __getattr__(name):
    if name in ("C2RFramework", "Qwen2_5VL"):
        return getattr(_models_module(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
