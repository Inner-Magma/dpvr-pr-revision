"""Method registry: maps method names to model classes.

Usage:
    @register_method("my_method")
    class MyModel(PreTrainedModel):
        @classmethod
        def from_config(cls, cfg): ...

    model = build_model(cfg)   # cfg["method"]["name"] == "my_method"
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel

_REGISTRY: dict[str, type] = {}


def register_method(name: str):
    """Class decorator that registers a model class under the given method name."""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def build_model(cfg: dict) -> "PreTrainedModel":
    """Instantiate the model for the method named in cfg['method']['name']."""
    name = cfg["method"]["name"]
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown method '{name}'. Registered methods: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name].from_config(cfg)


def list_methods() -> list[str]:
    return sorted(_REGISTRY)
