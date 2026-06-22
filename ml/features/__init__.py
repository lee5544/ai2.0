from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from typing import Any, Callable

DEFAULT_FEATURE_VERSION = "v14"
_FEATURE_MODULES: dict[str, tuple[str, str]] = {
    "v2": (".extract_features_v2", "extract_features_v2"),
    "v3": (".extract_features_v3", "extract_features_v3"),
    "v4": (".extract_features_v4", "extract_features_v4"),
    "v5": (".extract_features_v5", "extract_features_v5"),
    "v7": (".extract_features_v7", "extract_features_v7"),
    "v8": (".extract_features_v8", "extract_features_v8"),
    "v9": (".extract_features_v9", "extract_features_v9"),
    "v10": (".extract_features_v10", "extract_features_v10"),
    "v11": (".extract_features_v11", "extract_features_v11"),
    "v12": (".extract_features_v12", "extract_features_v12"),
    "v13": (".extract_features_v13", "extract_features_v13"),
    "v14": (".extract_features_v14", "extract_features_v14"),
}

SUPPORTED_FEATURE_VERSIONS: tuple[str, ...] = tuple(sorted(_FEATURE_MODULES))
LEGACY_FEATURE_VERSIONS: tuple[str, ...] = tuple(
    version for version in SUPPORTED_FEATURE_VERSIONS if version != DEFAULT_FEATURE_VERSION
)



def normalize_feature_version(version: str | None = None) -> str:
    normalized = str(version or DEFAULT_FEATURE_VERSION).strip().lower() or DEFAULT_FEATURE_VERSION
    if normalized not in _FEATURE_MODULES:
        supported = ", ".join(SUPPORTED_FEATURE_VERSIONS)
        raise ValueError(f"不支持的 feature_version: {normalized}，可选: {supported}")
    return normalized


def resolve_feature_version(cfg: dict[str, Any] | None = None) -> str:
    if isinstance(cfg, dict):
        direct_version = cfg.get("feature_version")
        if direct_version is not None:
            return normalize_feature_version(str(direct_version))

        features_cfg = cfg.get("features")
        if isinstance(features_cfg, dict):
            nested_version = features_cfg.get("version")
            if nested_version is not None:
                return normalize_feature_version(str(nested_version))

        train_cfg = cfg.get("train")
        if isinstance(train_cfg, dict):
            train_version = train_cfg.get("feature_version")
            if train_version is not None:
                return normalize_feature_version(str(train_version))

    return DEFAULT_FEATURE_VERSION


@lru_cache(maxsize=None)
def _load_feature_extractor(version: str) -> Callable[..., Any]:
    module_name, function_name = _FEATURE_MODULES[version]
    module = import_module(module_name, package=__name__)
    extractor = getattr(module, function_name, None)
    if not callable(extractor):
        raise AttributeError(f"特征提取函数不存在: {module_name}.{function_name}")
    return extractor


def get_feature_extractor(version: str | None = None) -> Callable[..., Any]:
    normalized = normalize_feature_version(version)
    return _load_feature_extractor(normalized)


def __getattr__(name: str) -> Any:
    for version, (_, function_name) in _FEATURE_MODULES.items():
        if name == function_name:
            return _load_feature_extractor(version)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_FEATURE_VERSION",
    "LEGACY_FEATURE_VERSIONS",
    "SUPPORTED_FEATURE_VERSIONS",
    "get_feature_extractor",
    "normalize_feature_version",
    "resolve_feature_version",
]
