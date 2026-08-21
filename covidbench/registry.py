"""Model registry with auto-discovery, so adding a model never edits a shared file."""
from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Callable

from . import config


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable
    features: tuple
    notes: str = ""
    tracks: tuple = (config.DEFAULT_TRACK,)


REGISTRY: dict[str, ModelSpec] = {}
UNAVAILABLE: dict[str, str] = {}
_loaded = False


def register(
    name: str,
    features: list[str] | None = None,
    notes: str = "",
    tracks: tuple | None = None,
):
    def decorator(factory: Callable):
        REGISTRY[name] = ModelSpec(
            name=name,
            factory=factory,
            features=tuple(features or config.FEATURES_ALL),
            notes=notes,
            tracks=tuple(tracks or (config.DEFAULT_TRACK,)),
        )
        return factory

    return decorator


def load_all() -> dict[str, ModelSpec]:
    """Import every module under models/. Missing optional deps are skipped, not fatal."""
    global _loaded
    if _loaded:
        return REGISTRY
    from . import models

    for module in pkgutil.iter_modules(models.__path__):
        try:
            importlib.import_module(f"{models.__name__}.{module.name}")
        except ImportError as exc:
            UNAVAILABLE[module.name] = str(exc)
    _loaded = True
    return REGISTRY


def get(name: str) -> ModelSpec:
    load_all()
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {', '.join(available())}")
    return REGISTRY[name]


def available() -> list[str]:
    load_all()
    return sorted(REGISTRY)


def available_for(track: str) -> list[str]:
    load_all()
    return sorted(name for name, spec in REGISTRY.items() if track in spec.tracks)


def features_for(spec: ModelSpec, track: str) -> list[str]:
    """Feature list a model uses on a given track.

    The inclusive track hands every model the two unknown indicators unless it already
    declares them, so a model does not need a separate registration per track.
    """
    features = list(spec.features)
    if track == config.COHORT_INCLUSIVE:
        features += [f for f in config.INDICATOR_FEATURES if f not in features]
    return features
