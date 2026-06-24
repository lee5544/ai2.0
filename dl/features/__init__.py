"""DL feature extraction package."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from typing import Any, Callable

DEFAULT_EXTRACTOR_NAME = "mel"
_EXTRACTOR_MODULES: dict[str, tuple[str, str]] = {
    "mel": (".extract_mel", "run"),
    "pcen": (".extract_pcen", "run"),
    "raw": (".extract_raw", "run"),
}

SUPPORTED_EXTRACTOR_NAMES: tuple[str, ...] = tuple(sorted(_EXTRACTOR_MODULES))


def normalize_extractor_name(name: str | None = None) -> str:
    normalized = str(name or DEFAULT_EXTRACTOR_NAME).strip().lower() or DEFAULT_EXTRACTOR_NAME
    alias_map = {
        "log_mel": "mel",
        "log-mel": "mel",
        "mel_spectrogram": "mel",
        "mel-spectrogram": "mel",
        "pcen_spectrogram": "pcen",
        "pcen-spectrogram": "pcen",
        "raw_signal": "raw",
        "raw-signal": "raw",
        "signal": "raw",
    }
    normalized = alias_map.get(normalized, normalized)
    if normalized not in _EXTRACTOR_MODULES:
        supported = ", ".join(SUPPORTED_EXTRACTOR_NAMES)
        raise ValueError(f"不支持的 DL 特征提取器: {normalized}，可选: {supported}")
    return normalized


def resolve_extractor_name(cfg: dict[str, Any] | None = None) -> str:
    if isinstance(cfg, dict):
        dl_cfg = cfg.get("dl")
        if isinstance(dl_cfg, dict):
            direct = dl_cfg.get("feature_type") or dl_cfg.get("feature_extractor")
            if direct is not None:
                return normalize_extractor_name(str(direct))

            extract_cfg = dl_cfg.get("extract")
            if isinstance(extract_cfg, dict):
                nested = (
                    extract_cfg.get("feature_type")
                    or extract_cfg.get("feature_extractor")
                    or extract_cfg.get("type")
                )
                if nested is not None:
                    return normalize_extractor_name(str(nested))

    return DEFAULT_EXTRACTOR_NAME


@lru_cache(maxsize=None)
def get_feature_extractor(name: str | None = None) -> Callable[..., Any]:
    normalized = normalize_extractor_name(name)
    module_name, function_name = _EXTRACTOR_MODULES[normalized]
    module = import_module(module_name, package=__name__)
    extractor = getattr(module, function_name, None)
    if not callable(extractor):
        raise AttributeError(f"DL 特征提取函数不存在: {module_name}.{function_name}")
    return extractor


def run(args: Any) -> None:
    from .extract_mel import _read_yaml

    cfg = _read_yaml(getattr(args, "config", None))
    cli_extractor = getattr(args, "feature_type", None) or getattr(args, "feature_extractor", None)
    extractor_name = normalize_extractor_name(cli_extractor) if cli_extractor else resolve_extractor_name(cfg)
    extractor = get_feature_extractor(extractor_name)
    extractor(args)


def parse_args() -> Any:
    from .extract_mel import parse_args as _parse_args

    return _parse_args()


def main() -> None:
    run(parse_args())


from .extract_mel import (  # noqa: E402
    COMPACT_FEATURE_COLUMN,
    DEFAULT_OUTPUT_FILE_PREFIX,
    DEFAULT_OUTPUT_FORMAT,
    SCHEMA_FILENAME,
    extract_mel_spectrogram_features,
)
from .extract_pcen import extract_pcen_spectrogram_features  # noqa: E402

__all__ = [
    "COMPACT_FEATURE_COLUMN",
    "DEFAULT_EXTRACTOR_NAME",
    "DEFAULT_OUTPUT_FILE_PREFIX",
    "DEFAULT_OUTPUT_FORMAT",
    "SCHEMA_FILENAME",
    "SUPPORTED_EXTRACTOR_NAMES",
    "extract_mel_spectrogram_features",
    "extract_pcen_spectrogram_features",
    "get_feature_extractor",
    "main",
    "normalize_extractor_name",
    "parse_args",
    "resolve_extractor_name",
    "run",
]
